/*** Calculating a derivative with CD ***/
#include <iostream>
#include <fstream>
#include <cmath>

void update (float *u, float *u_prev, int N, float h, float dt, float alpha)
{
	int I;
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = j*N+i; //converting to row major form 
			u_prev[I] = u[I];
		}
	}

	for (int j=1; j<N-1; j++)
	{	for (int i=1; i<N-1; i++) //for the interior nodes 
		{	I = j*N+i;
			u[I] = u_prev[I] + alpha*dt/(h*h) * (u_prev[I+1] + u_prev[I-1] + u_prev[I+N] + u_prev[I-N] - 4*u_prev[I]);
		}
	}

}

int main()
{
	// Allocate in CPU
	int N = 128; //128 mesh points

	float xmin 	= 0.0f;
	float xmax 	= 3.5f;
	float ymin 	= 0.0f;
	//float ymax 	= 2.0f;
	float h   	= (xmax-xmin)/(N-1);
	float dt	= 0.00001f;	
	float alpha	= 0.645f;
	float time 	= 0.4f;

	int steps = ceil(time/dt);
	int I;

//dynamically initialize NXN arrays 
	float *x  	= new float[N*N]; 
	float *y  	= new float[N*N]; 
	float *u  	= new float[N*N];
	float *u_prev  	= new float[N*N];


	// Generate mesh and intial condition
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
			x[I] = xmin + h*i;
			y[I] = ymin + h*j;
			u[I] = 0.0f;
			if ( (i==0) || (j==0)) //At lowest row and lowest column, temp =200, elsewhere 0 
				{u[I] = 200.0f;}
		}
	}

	// Loop 
	// double start = get_time();
	for (int t=0; t<steps; t++)
	{	update (u, u_prev, N, h, dt, alpha);
	}
	// double stop = get_time();
	
	// double elapsed = stop - start;
	// std::cout<<"time = "<<elapsed<<std::endl;

	std::ofstream temperature("temperature_cpu.txt");
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
		//	std::cout<<u[I]<<"\t";
			temperature<<x[I]<<" "<<y[I]<<" "<<u[I]<<std::endl;
		}
		// temperature<<"\n";
		//std::cout<<std::endl;
	}

	temperature.close();
}
