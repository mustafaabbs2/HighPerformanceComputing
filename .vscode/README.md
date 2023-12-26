# Server Implementations
## Mustafa Bhotvawala

# Build Instructions
1. Create a build/ directory in the root folder
2. Run the conan install script in pwsh (Windows) or .sh file for Linux/Darwin: this puts the find<>.cmake files in the folder
3. Run the build task CMake Configure for your platform and version. This runs cmake with the presets in the .json file in the root directory. 
4. Run the build task Build <release/debug>


# Dependencies
1. You need to run the conan install scripts in BuildScripts/
2. The conanfile.py specifies what is needed
3. Install conan 1.x if not available


# Sample

