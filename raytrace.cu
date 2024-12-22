#include <stdio.h>
#include <glm/glm.hpp>
#include <stdlib.h>
// #include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
// #define STB_IMAGE_RESIZE2_IMPLEMENTATION
// #include "stb/stb_image_resize2.h"


#define SSAA 2

#define IMAGE_WIDTH 2160
#define IMAGE_HEIGHT 1440

#define LIGHT_COUNT 2
#define BALL_COUNT 50

#define EAY_DEPTH 5.0f

#define BACKGROUND_COLOR glm::vec3(0.0f, 0.0f, 0.0f)

#define ENVIRONMENT_COLOR glm::vec3(1.0f, 1.0f, 1.0f)
#define KA 0.2f

struct light {
    glm::vec3 pos;
    glm::vec3 rgb;
    float strength;

    __device__ glm::vec3 get_ray_and_strength(const glm::vec3& o, float& str) const {
        glm::vec3 L = pos - o;
        float dist = glm::length(L);
        str = glm::min(strength / (dist * dist), 1.0f);
        return L;
    }
};

__constant__ light lights[LIGHT_COUNT];

struct ball {
    glm::vec3 pos;
    float radius;
    glm::vec3 rgb;
    float ka, kd, ks;
    float reflect_level;

    __device__ int hit(const glm::vec3& o, const glm::vec3& d, float& t, glm::vec3& normal) const {
        int inside = 0;
        glm::vec3 oc = pos - o;
        float l = glm::length(oc);
        float tp = glm::dot(oc, d);
        float dr = l * l - tp * tp;
        if(l < radius)//球体内
            inside = 1;
        else {
            if (tp <= 0.0f)//球在光线起点后
                return 0;
            if (dr > radius * radius) //光线不与球体相交
                return 0;
        }
        float thc = sqrtf(radius * radius - dr);
        if(inside)
            thc = -thc;
        if(tp - thc > t)
            return 0;
        t = tp - thc;
        normal = glm::normalize(d * t - oc);
        return 1;
    }
};

__constant__ ball balls[BALL_COUNT];

__global__ void render(unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;
    glm::vec3 o = glm::vec3(0.0f, 0.0f, -EAY_DEPTH);
    glm::vec3 d = glm::normalize(glm::vec3((float)(2 * x - width) / (float)height, (float)(2 * y - height)/ (float)height, EAY_DEPTH));
    float t = 1e9f;
    glm::vec3 normal;
    ball* hit_ball = nullptr;
    for (int i = 0; i < BALL_COUNT; i++) {
        if(balls[i].hit(o, d, t, normal))
            hit_ball = &balls[i];
    }
    if (hit_ball == nullptr) {
        image[index * 3 + 0] = (unsigned char)(BACKGROUND_COLOR[0] * 255.0);
        image[index * 3 + 1] = (unsigned char)(BACKGROUND_COLOR[1] * 255.0);
        image[index * 3 + 2] = (unsigned char)(BACKGROUND_COLOR[2] * 255.0);
        return;
    }
    ball &b = *hit_ball;
    glm::vec3 hit_pos = o + d * t;

    //环境光
    glm::vec3 ambient = b.ka * b.rgb * ENVIRONMENT_COLOR;

    //漫反射
    glm::vec3 diffuse = glm::vec3(0.0f, 0.0f, 0.0f);
    //镜面反射
    glm::vec3 specular = glm::vec3(0.0f, 0.0f, 0.0f);
    for(auto l : lights) {
        float str;
        glm::vec3 L = l.get_ray_and_strength(hit_pos, str);
        float dist = glm::length(L);
        L = glm::normalize(L);
        int shadow = 0;
        float t_tmp = 1e9f;
        glm::vec3 normal_tmp;
        for(int i = 0; i < BALL_COUNT; i++) {
            if(balls[i].hit(hit_pos + L * 0.001f, L, t_tmp, normal_tmp)){
                shadow = 1;
                break;
            }
        }
        if(shadow)
            continue;
        glm::vec3 R = glm::reflect(-L, normal);
        diffuse += str * glm::max(glm::dot(normal, L), 0.0f) * l.rgb;
        specular += str * glm::max(powf(glm::dot(R, -d), (int)b.reflect_level), 0.0f) * l.rgb;
    }
    diffuse *= b.kd * b.rgb;
    specular *= b.ks;


    glm::vec3 color = glm::clamp(ambient + diffuse + specular, 0.0f, 1.0f);
    image[index * 3 + 0] = (unsigned char)(color[0] * 255.0f);
    image[index * 3 + 1] = (unsigned char)(color[1] * 255.0f);
    image[index * 3 + 2] = (unsigned char)(color[2] * 255.0f);
}

__global__ void avg_pooling(unsigned char* oldimage, unsigned char* newimage, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width / SSAA + x;
    unsigned int r = 0;
    unsigned int g = 0;
    unsigned int b = 0;
    for (int i = 0; i < SSAA; i++) {
        for (int j = 0; j < SSAA; j++) {
            int x_ = x * SSAA + i;
            int y_ = y * SSAA + j;
            int index_ = y_ * width + x_;
            r += oldimage[index_ * 3 + 0];
            g += oldimage[index_ * 3 + 1];
            b += oldimage[index_ * 3 + 2];
        }
    }
    newimage[index * 3 + 0] = (unsigned char)(r / SSAA / SSAA);
    newimage[index * 3 + 1] = (unsigned char)(g / SSAA / SSAA);
    newimage[index * 3 + 2] = (unsigned char)(b / SSAA / SSAA);
}

int main() {
    int width = IMAGE_WIDTH * SSAA;
    int height = IMAGE_HEIGHT * SSAA;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    light* lights_host = (light*)malloc(sizeof(light) * LIGHT_COUNT);

    lights_host[0].pos = glm::vec3(-10.0f, 0.0f, 0.0f);
    lights_host[0].rgb = glm::vec3(1.0f, 1.0f, 1.0f);
    lights_host[0].strength = 400.0f;

    lights_host[1].pos = glm::vec3(0.0f, 5.0f, 0.0f);
    lights_host[1].rgb = glm::vec3(1.0f, 1.0f, 1.0f);
    lights_host[1].strength = 0.0f;

    cudaMemcpyToSymbol(lights, lights_host, sizeof(light) * LIGHT_COUNT);
    free(lights_host);

    ball* balls_host = (ball*)malloc(sizeof(ball) * BALL_COUNT);
    srand(time(NULL));
    for (int i = 0; i < BALL_COUNT; i++) {
        balls_host[i].rgb[0] = (float)rand() / RAND_MAX;
        balls_host[i].rgb[1] = (float)rand() / RAND_MAX;
        balls_host[i].rgb[2] = (float)rand() / RAND_MAX;
        balls_host[i].pos = glm::vec3((float)rand() / RAND_MAX * 4.0f - 2.0f, (float)rand() / RAND_MAX * 4.0f - 2.0f, (float)rand() / RAND_MAX * 10.0f);
        balls_host[i].radius = (float)rand() / RAND_MAX * 0.3f + 0.2f;
        balls_host[i].ka = KA;
        balls_host[i].ks = (float)rand() / RAND_MAX * 2.0f;
        balls_host[i].kd = 1 - (balls_host[i].ks / 2.0f) * 0.9f + 0.1f;
        balls_host[i].reflect_level = balls_host[i].ks / 2.0f * 180.0f + 20.0f;
    }
    cudaMemcpyToSymbol(balls, balls_host, sizeof(ball) * BALL_COUNT);
    free(balls_host);

    unsigned char* image;
    cudaMalloc(&image, width * height * 3);

    dim3 block_size(16, 16);
    dim3 grid_size(width / 16, height/ 16);
    render<<<grid_size, block_size>>>(image, width, height);
    cudaDeviceSynchronize();
    unsigned char* image_host = (unsigned char*)malloc(width * height * 3);
    cudaMemcpy(image_host, image, width * height * 3, cudaMemcpyDeviceToHost);

    int oh = IMAGE_HEIGHT;
    int ow = IMAGE_WIDTH;
    unsigned char* image_half;
    cudaMalloc(&image_half, ow * oh * 3);

    block_size = dim3(16, 16);
    grid_size = dim3(ow / 16, oh / 16);
    avg_pooling<<<grid_size, block_size>>>(image, image_half, width, height);
    cudaDeviceSynchronize();
    unsigned char* image_half_host = (unsigned char*)malloc(ow * oh * 3);
    cudaMemcpy(image_half_host, image_half, ow * oh * 3, cudaMemcpyDeviceToHost);

    cudaFree(image);
    cudaFree(image_half);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %.3f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    stbi_write_png("image.png", width, height, 3, image_host, width * 3);
    stbi_write_png("image_half.png", ow, oh, 3, image_half_host, ow * 3);
    free(image_host);
    free(image_half_host);
    return 0;
}

