#pragma once

void step_v0(float* r, const float* d, int n);
void step_v1(float* r, const float* d, int n);
void step_v2(float* r, const float* d, int n);
void step_v3(float* r, const float* d, int n);

#ifdef __GNUC__
void step_v4(float* r, const float* d, int n);
void step_v5(float* r, const float* d, int n);
#endif