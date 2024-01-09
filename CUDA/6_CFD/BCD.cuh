#pragma once

void update(float* u, float* u_prev, int N, float h, float dt, float alpha);
__global__ void update(float* u, float* u_prev, int N, float h, float dt, float alpha, int BSZ);
