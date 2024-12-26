#include <stdio.h>
#include <cmath>

#include <curand_kernel.h>

#include "glm/glm.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 640
#define SAMP 64
#define SUBPIX 4
#define MAX_DEPTH 6

#define MAX_BALL_COUNT 50

#define CUDAErrorCheck(ans) { cudaError_t error = ans; if (error != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(error)); exit(error); } }

// #define GENERATE_GAUSS_BLUR

class camera {
    int pixel_width;
    int pixel_height;
    float screen_width;
    float screen_height;
public:
    glm::vec3 pos;
    glm::vec3 screen_center;
    glm::vec3 up;

    camera(glm::vec3 p, glm::vec3 c, glm::vec3 u, int pw, int ph, float sw, float sh)
        : pos(p), screen_center(c), up(glm::normalize(u)), pixel_width(pw), pixel_height(ph), screen_width(sw), screen_height(sh) {}
    
    camera() {}

    __device__ glm::vec3 get_ray(float x, float y) const {
        glm::vec3 left = glm::normalize(glm::cross(up, screen_center - pos));
        glm::vec3 dir = glm::normalize(screen_center - pos - left* ((float)x / (float)pixel_width - 0.5f) * screen_width - up * ((float)y / (float)pixel_height - 0.5f) * screen_height);
        return dir;
    }
};

__constant__ camera cam;

/**
 * @brief cosine sample hemisphere
 * @param state - random state
 * @param normal - normal of the surface
 * @return cosine sample direction
 */
__device__ glm::vec3 cos_sample_hemisphere(curandStateXORWOW_t* state, const glm::vec3& normal) {
    double u1 = curand_uniform_double(state);
    double u2 = curand_uniform_double(state);
    
    double phi = 2.0 * M_PI * u1;
    double theta = std::acos(std::sqrt(1.0 - u2));

    double x = std::cos(phi) * std::sin(theta);
    double y = std::sin(phi) * std::sin(theta);
    double z = std::cos(theta);

    glm::vec3 local_dir = glm::vec3(x, y, z);
    
    return -glm::reflect(local_dir, glm::normalize(normal + glm::vec3(0, 0, 1)));
}

class item {
public:
    glm::vec3 rgb;
    glm::vec3 emission;

    __device__ int hit(const glm::vec3& o, const glm::vec3& d, float& t, glm::vec3& normal, int& inside);
    __device__ glm::vec3 brdf(curandStateXORWOW_t* state, const glm::vec3& wi, glm::vec3& wo, const glm::vec3& normal, const int& inside);
};

class Sphere : public item {
    glm::vec3 pos;
    float radius;
    float reflectance;//反射率
    float roughness;//粗糙度

    float refract_index;//折射率

    float specular;//反射占比
    float diffuse;//散射占比
    float refact;//折射占比

public:
    glm::vec3 rgb;
    glm::vec3 emission;

    Sphere(glm::vec3 p, float r, glm::vec3 rgb, float ref, float rou, glm::vec3 ems) 
        : pos(p), radius(r), rgb(rgb), reflectance(ref), roughness(rou), emission(ems) {
            specular = reflectance * (1.0 - roughness);
            diffuse = reflectance * roughness;
            refact = 1.0 - reflectance;
    }

    Sphere() {}

    __device__ int hit(const glm::vec3& o, const glm::vec3& d, float& t, glm::vec3& normal, int& inside) const {
        int ins = 0;
        glm::vec3 oc = pos - o;
        float l = glm::length(oc);
        float tp = glm::dot(oc, d);
        float dr = l * l - tp * tp;
        if(l < radius)//球体内
            ins = 1;
        else {
            if (tp <= 0.0f)//球在光线起点后
                return 0;
            if (dr > radius * radius) //光线不与球体相交
                return 0;
        }
        float thc = sqrtf(radius * radius - dr);
        if(ins)
            thc = -thc;
        if(tp - thc > t)
            return 0;
        t = tp - thc;
        normal = glm::normalize(d * t - oc);
        assert(normal.x == normal.x && normal.y == normal.y && normal.z == normal.z);
        if(ins)
            normal = -normal;
        inside = ins;
        return 1;
    }

    // lambertian diffuse
    __device__ glm::vec3 lambertian_brdf(curandStateXORWOW_t* state, const glm::vec3 &wi, glm::vec3& wo, const glm::vec3& normal, const int &inside) const {
        wo = cos_sample_hemisphere(state, normal);
        return rgb;
    }

    // specular
    __device__ glm::vec3 specular_brdf(curandStateXORWOW_t* state, const glm::vec3 &wi, glm::vec3& wo, const glm::vec3& normal, const int &inside) const {
        wo = glm::reflect(wi, normal);
        return rgb;
    }

    // refraction
    __device__ glm::vec3 refraction_brdf(curandStateXORWOW_t* state, const glm::vec3 &wi, glm::vec3& wo, const glm::vec3& normal, const int &inside) const {
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }

    /**
     * @brief mixture of brdf, set the outgoing direction and return the color
     * @param state - random state
     * @param wi - incident direction
     * @param wo - outgoing direction(will be set by this function)
     * @param normal - normal of the surface
     * @param inside - whether the surface is inside the object
     * @return cos(theta) * fr / pdf
    */
    __device__ glm::vec3 brdf(curandStateXORWOW_t* state, const glm::vec3 &wi, glm::vec3& wo, const glm::vec3& normal, const int &inside) const {
        float r1 = curand_uniform_double(state);
        if(r1 < refact)
            return refraction_brdf(state, wi, wo, normal, inside);
        else if (r1 < refact + diffuse)
            return lambertian_brdf(state, wi, wo, normal, inside);
        else
            return specular_brdf(state, wi, wo, normal, inside);
    }
};

__constant__ Sphere spheres[MAX_BALL_COUNT];
__constant__ int sphere_count;

/**
 * @brief shade for ray
 * @param state - random state
 * @param pos - ray origin
 * @param ray - ray direction
 * @param depth - ray depth
 * @return color
 */
__device__ glm::vec3 shade(curandStateXORWOW_t* state, const glm::vec3& pos, const glm::vec3& ray, int depth){
    glm::vec3 now_pos = pos;
    glm::vec3 now_ray = ray;

    glm::vec3 col(0.0f, 0.0f, 0.0f);
    glm::vec3 ratio(1.0f, 1.0f, 1.0f);

    for(int i = depth; i >=0; i--){
        float t = 1e30f;
        glm::vec3 normal;
        int inside;

        Sphere* hit_sphere = nullptr;
        for(int i = 0; i < sphere_count; i++) {
            if(spheres[i].hit(now_pos, now_ray, t, normal, inside))
                hit_sphere = &spheres[i];
        }
        if(hit_sphere == nullptr)
            return col;
        Sphere& s = *hit_sphere;
        if(depth <= 0)
            return col + ratio * s.emission;
        now_pos = now_pos + now_ray * t + 0.03f * normal;
        col += ratio * s.emission;
        ratio *= s.brdf(state, now_ray, now_ray, normal, inside);
    }
    return col;
}

/**
 * @brief init random states for threads
 * @param states - random states
 */
__global__ void init_states(curandStateXORWOW_t* states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * IMAGE_WIDTH + x;
    if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT) return;
    curand_init(1234, index, 0, &states[index]);
}

/**
 * @brief render image
 * @param states - random states
 * @param image - output image
 * @param width - image width
 * @param height - image height
 * @param samples_per_pixel - number of samples per pixel
 */
__global__ void render(curandStateXORWOW_t* states, float* image, int width, int height, int samples_per_pixel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;
    if (x >= width || y >= height) return;

    glm::vec3 col(0.0f, 0.0f, 0.0f);
    glm::vec3 pos = cam.pos;

    //每个像素采样多个像素
    for(int i = 0; i < SUBPIX; i++) {
        assert(x < width && y < height);
        float theta = curand_uniform_double(&states[index]) * 2.0 * M_PI;
        assert(index < width * height);
        float r = curand_uniform_double(&states[index]);
        glm::vec3 samp_ray = cam.get_ray(x + r * std::cos(theta), y + r * std::sin(theta));
        col += shade(&states[index], pos, samp_ray, MAX_DEPTH) / (float)samples_per_pixel / (float)SUBPIX;
    }
    image[index * 3 + 0] += col[0];
    image[index * 3 + 1] += col[1];
    image[index * 3 + 2] += col[2];
}

#ifdef GENERATE_GAUSS_BLUR
__constant__ unsigned char gauss_kernel[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

/**
 * @brief apply gaussian blur to image
 * @param newimage - new image after blur
 * @param oldimage - old image before blur
 * @param width - image width
 * @param height - image height
 */
__global__ void gauss_blur(unsigned char* newimage, unsigned char* oldimage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    unsigned int r = 0, g = 0, b = 0;
    for(int i = 0; i <= 2; i++) {
        for(int j = 0; j <= 2; j++) {
            int x_ = x + i - 1;
            int y_ = y + j - 1;
            if(x_ < 0 || x_ >= width || y_ < 0 || y_ >= height)
                continue;
            int index_ = y_ * width + x_;
            r += oldimage[index_ * 3 + 0] * gauss_kernel[i][j];
            g += oldimage[index_ * 3 + 1] * gauss_kernel[i][j];
            b += oldimage[index_ * 3 + 2] * gauss_kernel[i][j];
        }
    }
    newimage[index * 3 + 0] = (unsigned char)(r / 16);
    newimage[index * 3 + 1] = (unsigned char)(g / 16);
    newimage[index * 3 + 2] = (unsigned char)(b / 16);
}
#endif

/**
 * @brief transfer float image to uint8 image
 * @param newimage - uint8 image
 * @param oldimage - float image
 * @param width - image width
 * @param height - image height
 */
__global__ void image_float2uint8(unsigned char* newimage, float* oldimage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;
    if (x >= width || y >= height) return;
    glm::vec3 rgb = glm::clamp(glm::vec3(oldimage[index * 3 + 0], oldimage[index * 3 + 1], oldimage[index * 3 + 2]), 0.0f, 1.0f);
    glm::vec3 rgb8 = 255.99f * rgb;
    newimage[index * 3 + 0] = (unsigned char)rgb8[0];
    newimage[index * 3 + 1] = (unsigned char)rgb8[1];
    newimage[index * 3 + 2] = (unsigned char)rgb8[2];
}

/**
 * @brief init camera
 */
void init_camera(){
    camera cam_host = camera(
        glm::vec3(40.8, 40.8, 162), // eye
        glm::vec3(40.8, 40.8, 161.2), // see
        glm::vec3(0.0f, 1.0f, 0.0f), // up
        IMAGE_WIDTH, // pixel_width
        IMAGE_HEIGHT, // pixel_height
        (float)IMAGE_WIDTH / (float)IMAGE_HEIGHT, // screen_width
        1.0f // screen_height
    );
    cudaMemcpyToSymbol(cam, &cam_host, sizeof(camera));
}

/**
 * @brief init spheres
 */
void init_spheres(){
    Sphere spheres_host[] = {
        Sphere(glm::vec3(1e5, 40.8, 81.6), 1e5, glm::vec3(0.75, 0.25, 0.25), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // Left
        Sphere(glm::vec3(-1e5 + 81.6, 40.8, 81.6), 1e5, glm::vec3(0.25, 0.25, 0.75), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // Right
        Sphere(glm::vec3(40.8, 40.8, 1e5), 1e5, glm::vec3(0.75, 0.75, 0.75), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // Back
        Sphere(glm::vec3(40.8, 40.8, -1e5 + 163.2), 1e5, glm::vec3(0.0, 0.0, 0.0), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // Front
        Sphere(glm::vec3(40.8, 1e5, 81.6), 1e5, glm::vec3(0.75, 0.75, 0.75), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // Bottom
        Sphere(glm::vec3(40.8, -1e5 + 81.6, 81.6), 1e5, glm::vec3(0.75, 0.75, 0.75), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // Top
        Sphere(glm::vec3(27, 16, 48), 16, glm::vec3(1.0, 1.0, 1.0), 1.0, 0.8, glm::vec3(0.0, 0.0, 0.0)), // sphere1
        Sphere(glm::vec3(56, 16, 74), 16, glm::vec3(0.10, 0.7, 0.10), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // sphere2
        Sphere(glm::vec3(40.8, 681.6 - 0.16, 81.6), 600, glm::vec3(0.0, 0.0, 0.0), 1.0, 1.0, glm::vec3(24, 24, 24)) // Light
    };
    int sphere_count_host = sizeof(spheres_host) / sizeof(Sphere);
    cudaMemcpyToSymbol(spheres, spheres_host, sizeof(spheres_host));
    cudaMemcpyToSymbol(sphere_count, &sphere_count_host, sizeof(int));
}

int main() {
    int width = IMAGE_WIDTH;
    int height = IMAGE_HEIGHT;
    int samples_per_pixel = SAMP;

    cudaEvent_t start, stop;
    CUDAErrorCheck(cudaEventCreate(&start));
    CUDAErrorCheck(cudaEventCreate(&stop));
    CUDAErrorCheck(cudaEventRecord(start, 0));

    init_camera();
    init_spheres();

    // set up CUDA threads and blocks
    dim3 block_size(16, 16);
    dim3 grid_size((width + 15) / 16, (height + 15)/ 16);

    // init random states
    curandStateXORWOW_t* states;
    size_t size = width * height;
    CUDAErrorCheck(cudaMalloc(&states, sizeof(curandStateXORWOW_t) * size));
    init_states<<<grid_size, block_size>>>(states);

    // render
    float* image_f;
    CUDAErrorCheck(cudaMalloc(&image_f, width * height * 3 * sizeof(float)));
    CUDAErrorCheck(cudaMemset(image_f, 0, width * height * 3 * sizeof(float)));
    for(int i = 0; i < samples_per_pixel; i++){
        render<<<grid_size, block_size>>>(states, image_f, width, height, samples_per_pixel);
        if(i % 10 == 9){
            fflush(stderr);
            fprintf(stderr, "Rendering... %5.2f%%\r", (float)(i+1) / (float)samples_per_pixel * 100.0f);
        }
        cudaDeviceSynchronize();
    }
    CUDAErrorCheck(cudaGetLastError());

    // convert float(0.0~1.0) to uint8(0~255)
    unsigned char* image;
    CUDAErrorCheck(cudaMalloc(&image, width * height * 3));
    image_float2uint8<<<grid_size, block_size>>>(image, image_f, width, height);
    cudaDeviceSynchronize();
    CUDAErrorCheck(cudaGetLastError());
    cudaFree(image_f);

#ifdef GENERATE_GAUSS_BLUR
    // blur image with gaussian kernel
    unsigned char* gauss_image;
    CUDAErrorCheck(cudaMalloc(&gauss_image, width * height * 3));
    CUDAErrorCheck(cudaMemset(gauss_image, 0, width * height * 3));
    gauss_blur<<<grid_size, block_size>>>(gauss_image, image, width, height);
    cudaDeviceSynchronize();
    CUDAErrorCheck(cudaGetLastError());

    // copy result to host
    unsigned char* gauss_image_host = (unsigned char*)malloc(width * height * 3);
    CUDAErrorCheck(cudaMemcpy(gauss_image_host, gauss_image, width * height * 3, cudaMemcpyDeviceToHost));
    CUDAErrorCheck(cudaFree(gauss_image));
#endif
    // copy result to host
    unsigned char* image_host = (unsigned char*)malloc(width * height * 3);
    CUDAErrorCheck(cudaMemcpy(image_host, image, width * height * 3, cudaMemcpyDeviceToHost));
    CUDAErrorCheck(cudaFree(image));

    // get and print time
    CUDAErrorCheck(cudaEventRecord(stop, 0));
    CUDAErrorCheck(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDAErrorCheck(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time: %.3f ms         \n", milliseconds);
    CUDAErrorCheck(cudaEventDestroy(start));
    CUDAErrorCheck(cudaEventDestroy(stop));

    // save image to file
    stbi_write_png("image.png", width, height, 3, image_host, width * 3);
    free(image_host);
#ifdef GENERATE_GAUSS_BLUR
    stbi_write_png("gauss_image.png", width, height, 3, gauss_image_host, width * 3);
    free(gauss_image_host);
#endif
    return 0;
}

