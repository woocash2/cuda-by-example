#include <iostream>
using namespace std;

// __FILE__ is a preprocessor macro that expands to full path to the current file
// __LINE__ is a preprocessor macro that expands to current line number in the source file

#define errchk(e) errchk_(e, __FILE__, __LINE__)

// Funkcja sprawdzająca czy wykonanie funkcji z zakresu bibliotek CUDA się powiodło i wypisuje odpowiedni błąd
// oraz miejsce w kodzie, które spowodowało błąd
__host__ void errchk_(cudaError_t e, const char * file, int line) {
    if (e != cudaSuccess) {
        cerr << file << ": " << line << ": " << cudaGetErrorString(e) << endl;
        exit(e);
    }
}