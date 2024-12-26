#include <stdio.h>
#include <cmath>
#include <stdlib.h>

#include <curand_kernel.h>

#include "glm/glm.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define IMAGE_WIDTH 1
#define IMAGE_HEIGHT 1
#define SAMP 32
#define SUBPIX 4
#define MAX_DEPTH 8

#define MAX_BALL_COUNT 50

#define CUDAErrorCheck(ans) { cudaError_t error = ans; if (error != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(error)); exit(error); } }

// #define GENERATE_GAUSS

struct camera {
    glm::vec3 pos;
    glm::vec3 screen_center;
    glm::vec3 up;
    int pixel_width;
    int pixel_height;
    float screen_width;
    float screen_height;
    
    __device__ glm::vec3 get_ray(float x, float y) const {
        glm::vec3 left = glm::normalize(glm::cross(up, screen_center - pos));
        glm::vec3 dir = glm::normalize(screen_center - pos - left* ((float)x / (float)pixel_width - 0.5f) * screen_width - up * ((float)y / (float)pixel_height - 0.5f) * screen_height);
        return dir;
    }
};

__constant__ camera cam;

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

struct Sphere {
    glm::vec3 pos;
    float radius;
    glm::vec3 rgb;
    float reflectance;//反射率
    float roughness;//粗糙度
    glm::vec3 emission;

    float specular;//反射占比
    float diffuse;//散射占比
    float refact;//折射占比

    Sphere(glm::vec3 p, float r, glm::vec3 rgb, float ref, float rou, glm::vec3 ems) 
        : pos(p), radius(r), rgb(rgb), reflectance(ref), roughness(rou), emission(ems) {
            specular = reflectance * (1.0f - roughness);
            diffuse = reflectance * roughness;
            refact = 1 - reflectance;
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
        return rgb * diffuse;
    }

    // specular
    __device__ glm::vec3 specular_brdf(curandStateXORWOW_t* state, const glm::vec3 &wi, glm::vec3& wo, const glm::vec3& normal, const int &inside) const {
        wo = glm::reflect(wi, normal);
        return rgb * specular;
    }

    // refraction
    __device__ glm::vec3 refraction_brdf(curandStateXORWOW_t* state, const glm::vec3 &wi, glm::vec3& wo, const glm::vec3& normal, const int &inside) const {
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }

    /**
     * @brief mixture of brdf
     * @return fr / pdf
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

__device__ glm::vec3 shade(curandStateXORWOW_t* state, const glm::vec3& pos, const glm::vec3& ray, int depth){
    float t = 1e30f;
    glm::vec3 normal;
    int inside;
    if (depth == 3)
        printf ("depth == 3\n");
    Sphere* hit_sphere = nullptr;
    for(int i = 0; i < sphere_count; i++) {
        if(spheres[i].hit(pos, ray, t, normal, inside))
            hit_sphere = &spheres[i];
    }
    if(hit_sphere == nullptr){
        printf("No hit\n");
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }
    Sphere& s = *hit_sphere;
    if(depth <= 0)
        return s.emission;

    glm::vec3 wo;
    glm::vec3 brdf = s.brdf(state, ray, wo, normal, inside);
    glm::vec3 hit_pos = pos + ray * t + 0.03f * normal;

    printf("depth : %d\n", depth);
    printf("hit_pos: %f %f %f\n", hit_pos.x, hit_pos.y, hit_pos.z);
    printf("wo: %f %f %f\n", wo.x, wo.y, wo.z);
    auto result = shade(state, hit_pos, wo, depth - 1);
    printf("result: %f %f %f\n", result.x, result.y, result.z);
    return s.emission + brdf * result;
}

__global__ void render(curandStateXORWOW_t* states, unsigned char* image, int width, int height, int samples_per_pixel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;
    if (x >= width || y >= height) return;
    curand_init(1234, index, 0, &states[index]);

    glm::vec3 col(0.0f, 0.0f, 0.0f);
    
    glm::vec3 pos = cam.pos;
    // glm::vec3 ray = cam.get_ray(x, y);

    //采样 4 个像素
    for(int s = 0; s < samples_per_pixel; s++) {
        for(int i = 0; i < SUBPIX; i++) {
            assert(x < width && y < height);
            float theta = curand_uniform_double(&states[index]) * 2.0 * M_PI;
            assert(index < width * height);
            float r = curand_uniform_double(&states[index]);
            glm::vec3 samp_ray = cam.get_ray(x + r * std::cos(theta), y + r * std::sin(theta));
            col += shade(&states[index], pos, samp_ray, MAX_DEPTH) / (float)samples_per_pixel / (float)SUBPIX;
        }
    }

    // printf("aaaaaa");


    glm::vec3 rgb = 255.99f * glm::clamp(col, 0.0f, 1.0f);

    image[index * 3 + 0] = (unsigned char)(rgb[0]);
    image[index * 3 + 1] = (unsigned char)(rgb[1]);
    image[index * 3 + 2] = (unsigned char)(rgb[2]);
}

#ifdef GENERATE_GAUSS
__constant__ unsigned char gauss_kernel[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

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

void init_camera(){
    camera cam_host = {
        glm::vec3(40.8, 40.8, 162),
        glm::vec3(40.8, 40.8, 161.3),
        glm::vec3(0.0f, 1.0f, 0.0f),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        (float)IMAGE_WIDTH / (float)IMAGE_HEIGHT,
        1.0f
    };
    cudaMemcpyToSymbol(cam, &cam_host, sizeof(camera));
}

void init_spheres(){
    Sphere spheres_host[] = {
        Sphere(glm::vec3(1e5, 40.8, 81.6), 1e5, glm::vec3(0.75, 0.25, 0.25), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // Left
        Sphere(glm::vec3(-1e5 + 81.6, 40.8, 81.6), 1e5, glm::vec3(0.25, 0.25, 0.75), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // Right
        Sphere(glm::vec3(40.8, 40.8, 1e5), 1e5, glm::vec3(0.75, 0.75, 0.75), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // Back
        Sphere(glm::vec3(40.8, 40.8, -1e5 + 163.2), 1e5, glm::vec3(0.0, 0.0, 0.0), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // Front
        Sphere(glm::vec3(40.8, 1e5, 81.6), 1e5, glm::vec3(0.75, 0.75, 0.75), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // Bottom
        Sphere(glm::vec3(40.8, -1e5 + 81.6, 81.6), 1e5, glm::vec3(0.75, 0.75, 0.75), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // Top
        Sphere(glm::vec3(27, 16, 48), 16, glm::vec3(1.0, 1.0, 1.0), 1.0, 0.8, glm::vec3(0.0, 0.0, 0.0)), // sphere1
        Sphere(glm::vec3(56, 16, 74), 16, glm::vec3(0.10, 1.0, 0.10), 1.0, 1.0, glm::vec3(0.0, 0.0, 0.0)), // sphere2
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
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    init_camera();

    init_spheres();

    unsigned char* image;
    CUDAErrorCheck(cudaMalloc(&image, width * height * 3));

    curandStateXORWOW_t* states;
    size_t size = width * height;
    CUDAErrorCheck(cudaMalloc(&states, sizeof(curandStateXORWOW_t) * size));

    dim3 block_size(16, 16);
    dim3 grid_size((width + 15) / 16, (height + 15)/ 16);
    render<<<grid_size, block_size>>>(states, image, width, height, samples_per_pixel);
    cudaDeviceSynchronize();
    CUDAErrorCheck(cudaGetLastError());

#ifdef GENERATE_GAUSS
    unsigned char* gauss_image;
    cudaMalloc(&gauss_image, width * height * 3);

    dim3 block_size_gauss(16, 16);
    dim3 grid_size_gauss(width / 16, height/ 16);
    gauss_blur<<<grid_size_gauss, block_size_gauss>>>(gauss_image, image, width, height);
    cudaDeviceSynchronize();


    unsigned char* gauss_image_host = (unsigned char*)malloc(width * height * 3);
    cudaMemcpy(gauss_image_host, gauss_image, width * height * 3, cudaMemcpyDeviceToHost);
    cudaFree(gauss_image);
#endif
    unsigned char* image_host = (unsigned char*)malloc(width * height * 3);
    cudaMemcpy(image_host, image, width * height * 3, cudaMemcpyDeviceToHost);
    cudaFree(image);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %.3f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    stbi_write_png("image.png", width, height, 3, image_host, width * 3);
    free(image_host);
#ifdef GENERATE_GAUSS
    stbi_write_png("gauss_image.png", width, height, 3, gauss_image_host, width * 3);
    free(gauss_image_host);
#endif
    return 0;
}

