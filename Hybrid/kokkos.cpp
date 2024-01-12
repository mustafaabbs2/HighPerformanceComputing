#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

#include "kokkos.h"
namespace Core
{
struct hello_world
{
	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const
	{
		printf("Hello from i = %i\n", i);
	}
};

void init()
{

	Kokkos::initialize();
}

void finalize()
{
	Kokkos::finalize();
}

void helloWorld()
{

	printf("Hello World on Kokkos execution space %s\n",
		   typeid(Kokkos::DefaultExecutionSpace).name());

	Kokkos::parallel_for("HelloWorld", 15, hello_world());
}

} // namespace Core