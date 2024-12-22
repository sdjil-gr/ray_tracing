#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <glm/glm.hpp>
// #include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
// #define STB_IMAGE_RESIZE2_IMPLEMENTATION
// #include "stb/stb_image_resize2.h"

#define IMAGE_WIDTH 4320
#define IMAGE_HEIGHT 2880

#define BALL_COUNT 50

#define EAY_DEPTH 10.0f

struct ball {
    float r, g, b;
    glm::vec3 pos;
    float radius;

    int hit(const glm::vec3& o, const glm::vec3& d, float& t, glm::vec3& normal) const {
        glm::vec3 oc = o - pos;
        float a = glm::dot(d, d);
        float b = 2.0f * glm::dot(d, oc);
        float c = glm::dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {
            return 0;
        }
        float sqrt_discriminant = sqrt(discriminant);
        float t1 = (-b - sqrt_discriminant) / (2 * a);
        float t2 = (-b + sqrt_discriminant) / (2 * a);
        if (t1 > t2) {
            float temp = t1;
            t1 = t2;
            t2 = temp;
        }
        if (t1 > 0.001f && t1 < t) {
            t = t1;
            normal = (o + d * t1 - pos) / radius;
            return 1;
        }
        if (t2 > 0.001f && t2 < t) {
            t = t2;
            normal = (o + d * t2 - pos) / radius;
            return 1;
        }
        return 0;
    }
};

ball balls[BALL_COUNT];

void render(unsigned char* image, int width, int height, int x, int y) {
    int index = y * width + x;
    glm::vec3 o = glm::vec3(0.0f, 0.0f, -EAY_DEPTH);
    glm::vec3 d = glm::normalize(glm::vec3((float)(2 * x - width) / (float)height, (float)(2 * y - height)/ (float)height, EAY_DEPTH));
    float t = 1e9f;
    glm::vec3 normal;
    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;
    for (int i = 0; i < BALL_COUNT; i++) {
        if(balls[i].hit(o, d, t, normal)){
            float fcos = -glm::dot(normal, d);
            r = (unsigned char)(balls[i].r * fcos * 255);
            g = (unsigned char)(balls[i].g * fcos * 255);
            b = (unsigned char)(balls[i].b * fcos * 255);
        }
    }
    image[index * 3 + 0] = r;
    image[index * 3 + 1] = g;
    image[index * 3 + 2] = b;
}

void avg_2x2pooling(unsigned char* oldimage, unsigned char* newimage, int width, int height, int x, int y){
    int index = y * width / 2 + x;
    unsigned int r = 0;
    unsigned int g = 0;
    unsigned int b = 0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int x_ = x * 2 + i;
            int y_ = y * 2 + j;
            int index_ = y_ * width + x_;
            r += oldimage[index_ * 3 + 0];
            g += oldimage[index_ * 3 + 1];
            b += oldimage[index_ * 3 + 2];
        }
    }
    newimage[index * 3 + 0] = (unsigned char)(r / 4);
    newimage[index * 3 + 1] = (unsigned char)(g / 4);
    newimage[index * 3 + 2] = (unsigned char)(b / 4);
}

int main() {
    int width = IMAGE_WIDTH;
    int height = IMAGE_HEIGHT;

    //获取毫秒时间
    clock_t start_time = clock();

    srand(time(NULL));
    for (int i = 0; i < BALL_COUNT; i++) {
        balls[i].r = (float)rand() / RAND_MAX;
        balls[i].g = (float)rand() / RAND_MAX;
        balls[i].b = (float)rand() / RAND_MAX;
        balls[i].pos = glm::vec3((float)rand() / RAND_MAX * 4.0f - 2.0f, (float)rand() / RAND_MAX * 4.0f - 2.0f, (float)rand() / RAND_MAX * 10.0f);
        balls[i].radius = (float)rand() / RAND_MAX * 0.4f + 0.10f;
    }

    unsigned char* image = (unsigned char*)malloc(width * height * 3);
    for(int i = 0; i < width; i++)
        for(int j = 0; j < height; j++)
            render(image, width, height, i, j);

    int oh = height / 2;
    int ow = width / 2;
    unsigned char* image_half = (unsigned char*)malloc(ow * oh * 3);
    for(int i = 0; i < ow; i++)
        for(int j = 0; j < oh; j++)
            avg_2x2pooling(image, image_half, width, height, i, j);
    
    clock_t end_time = clock();
    printf("Time: %.3f ms\n", (double)(end_time - start_time) * 1000 / CLOCKS_PER_SEC);

    stbi_write_png("image(cpu).png", width, height, 3, image, width * 3);
    stbi_write_png("image_half(cpu).png", ow, oh, 3, image_half, ow * 3);
    free(image);
    free(image_half);
    return 0;
}

