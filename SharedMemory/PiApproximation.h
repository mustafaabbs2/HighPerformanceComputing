#pragma once

void warmup();
double approximatePiSerial(size_t numSteps);
double approximatePiParallel(size_t numSteps);
double approximatePiParallelNoReduction(size_t numSteps);
double approximatePiParallelPadded(size_t numSteps);
double approximatePiStdPar(size_t numSteps);
double approximatePiParallelThreads(size_t numSteps, int numThreads);
