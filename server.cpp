#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

using namespace std;

void error_handling(char* str){
    cout<<str<<endl;
    exit(-1);
}

int main(){
    const int BUF_SIZE = 512*512;
    char message[BUF_SIZE];
    int str_len;
    const int PORT = 3000;
    struct sockaddr_in serv_addr;
    struct sockaddr_in clnt_addr;

    int serv_sock = socket(PF_INET,SOCK_STREAM,0);
    if(serv_sock == -1)
        error_handling("socket() error");
    cout<<"socket()"<<endl;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port=htons(PORT);
    
    if(bind(serv_sock,(struct sockaddr*)&serv_addr,sizeof(serv_addr))==-1)
        error_handling("bind() error");
    cout<<"bind()"<<endl;
    if(listen(serv_sock,1)==-1)
        error_handling("listen() error");
    cout<<"listen()"<<endl;
    socklen_t clnt_addr_size = sizeof(clnt_addr);
    int clnt_sock = accept(serv_sock,(struct sockaddr*)&clnt_addr,&clnt_addr_size);
    if(clnt_sock==-1)
        error_handling("accept() error");
    cout<<"Connected"<<endl;
    
    str_len = read(clnt_sock,message,BUF_SIZE);
    if(str_len == -1) error_handling("read() error");
    cout<<"str_len : "<<str_len<<endl;
    cout<<"message : "<<message<<endl;
    int result = write(clnt_sock,message,str_len);
    if(result == -1) error_handling("write() error");
    
    cout<<"close"<<endl;
    close(serv_sock);
    close(clnt_sock);
    return 0;
}

/*
//소켓 생성
int socket(int domain,int type,int protocol); //성공시 파일 디스크립터, 실패시 -1
//domain : PF_INET
//type : SOCK_STREAM
//protocol : IPPRORO_TCP;


//주소 할당
int bind(int sockfd,struct sockaddr *myaddr,socklen_t addrlen); //성공시 0, 실패시 -1
//연결가능 상태로 변경
int listen(int sockfd,int backlog); //성공시 0, 실패시 -1
//연결요청 가능 상태로 변경
 int accept(int sockfd,struct sockaddr *addr,socklen_t *addrlen)//성공시 파일 디스크립터, 실패시 -1
//연경 요청
int connect(int sockfd, struct sockaddr *serv_addr,socklen_t addrlen);// 성공시 0, 실패시 -1
*/
