echo Compiling...
echo nvcc -lcuda -diag-suppress 20012 raytrace.cu -o raytrace
nvcc -lcuda -diag-suppress 20012 raytrace.cu -o raytrace
echo Compiled.
echo Running...
./raytrace
echo Done.