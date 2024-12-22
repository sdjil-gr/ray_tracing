echo Compiling...
echo    nvcc -O3 -lcuda -diag-suppress 20012 raytrace.cu -o raytrace
nvcc -O3 -lcuda -diag-suppress 20012 raytrace.cu -o raytrace
echo Compiled.
echo Running...
./raytrace
echo Done.