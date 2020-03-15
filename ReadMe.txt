Building the project:
- Have the CUDA Toolkit installed for Visual Studio (I used Visual Studio 2017)
- Once in the solution make sure the imported files are pathed correctly to your machine.
- If everything is properly installed and pathed, you should be able to build the solution
- The release exe for x64 will be in the path x64/Release/GameOfLife.exe

Running the project:
- To run, run the exe in the path x64/Release/GameOfLife.exe
- Note, you must have a CUDA compatable GPU installed
- The program will run a serial CPU version of Conway's Game Of Life first, then parallel CUDA
- Each will run for 10 seconds, running as many iterations of the game it can during that time.
- The output of the game will be printed to a file called output.txt which will be located in the directory of the exe
- The input data for both runs is random, but the same for both

Performance on my machine:
- World size of 10,000
- Serial: Around 300,000 iterations 
- Parallel: Around 4,550,000 iterations (at 1000 threads)
- Machine specs:
	- Intel i7 8086 @ 4.4 GHz
	- Nvidia GTX 1080 @ 2062 MHz

Notes about implementation:
- Original implementation used if statements
- Original implementation world was not cyclic, edges outside of the game board were always considered dead
- Based off of NightElfik's implementation, I added a cyclic world, used the idea of the lifeData being an unsigned char,
	and made variables larger to run larger world sizes and threads
- If threads goes above 1000 even with a very large world size, the final output of the parallel run will be blank
    - I am unsure if this is my implemntation's fault or that it reaches my GPU's limit