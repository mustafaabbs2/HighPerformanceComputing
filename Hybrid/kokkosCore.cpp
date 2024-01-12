#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

#include "KokkosCore.h"
namespace Hybrid
{
struct hello_world
{
	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const
	{
		printf("Hello from i = %i\n", i);
	}
};

void helloWorld()
{
	Kokkos::initialize();

	printf("Hello World on Kokkos execution space %s\n",
		   typeid(Kokkos::DefaultExecutionSpace).name());

	Kokkos::parallel_for("HelloWorld", 15, hello_world());

	Kokkos::finalize();
}

} // namespace Hybrid