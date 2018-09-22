#include<cstdio>
#include<cstdlib>
#include<cstring>
#define N 10

int main(){
    char *arr[N];
    char tmp[256];
    for(int i=0;i<N;i++){
        scanf("%s",tmp);
        arr[i] = (char *)malloc(sizeof(strlen(tmp)+1));
        strcpy(arr[i],tmp);
    }
    puts("");
    for(int i=0;i<N;i++){
        printf("%s\n",arr[i]);
        free(arr[i]);
    }
    return 0;
    
}
