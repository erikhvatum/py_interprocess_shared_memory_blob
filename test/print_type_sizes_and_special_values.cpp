#include <iostream>
#include <semaphore.h>

using namespace std;

int main(int, char**)
{
    cout << "typename: size_in_bytes" << endl;
    cout << "sem_t: " << sizeof(sem_t) << endl;
    cout << "MACRO or other special name: value" << endl;
    cout << "SEM_FAILED: " << SEM_FAILED << endl;
}