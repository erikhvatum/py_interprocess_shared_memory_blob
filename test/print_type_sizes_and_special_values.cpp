#include <iostream>
#include <pthread.h>
#include <semaphore.h>

using namespace std;

int main(int, char**)
{
    cout << "typename: size_in_bytes" << endl;
    cout << "sem_t: " << sizeof(sem_t) << endl;
    cout << "pthread_mutexattr_t: " << sizeof(pthread_mutexattr_t) << endl;
    cout << "pthread_rwlockattr_t: " << sizeof(pthread_rwlockattr_t) << endl;
    cout << "pthread_rwlock_t: " << sizeof(pthread_rwlock_t) << endl;
    cout << "MACRO or other special name: value" << endl;
    cout << "SEM_FAILED: " << SEM_FAILED << endl;
    cout << "PTHREAD_PROCESS_SHARED: " << PTHREAD_PROCESS_SHARED << endl;
}
