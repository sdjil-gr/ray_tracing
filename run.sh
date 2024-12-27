echo Compiling...
echo nvcc -O3 -lcuda -diag-suppress 20012 raytrace.cu -o raytrace -lcudart_static
nvcc -O3 -lcuda -diag-suppress 20012 raytrace.cu -o raytrace -lcudart_static
echo Compiled.
echo Running...
./raytrace
echo Done.