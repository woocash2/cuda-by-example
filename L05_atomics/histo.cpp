#include <iostream>
#include <string.h>
using namespace std;

const int SIZE = 1024 * 1024 * 100; // 100 MB

int main() {

    unsigned char * data = (unsigned char *) malloc(SIZE);
    for (int i = 0; i < SIZE; i++)
        data[i] = rand();
        
    int histo[256];
    memset(histo, 0, 256 * sizeof(int));
    for (int i = 0; i < SIZE; i++)
        histo[data[i]]++;
    
    int sum = 0;
    for (int i = 0; i < 256; i++)
        sum += histo[i];

    if (sum == SIZE)
        cout << "OK";
    else
        cout << "Wrong: " << sum << " " << SIZE;

    free(data);
    return 0;
}