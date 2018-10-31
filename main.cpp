#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstring>
#include <fstream>
#include <cstdlib>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>


using namespace std;

void error_handling(char* str){
    cout<<str<<endl;
    exit(-1);
}

cv::Mat masking(cv::Mat, int, int);//return masking image
int boundary(int *, int *);
void fill(cv::Mat);
int * find(cv::Mat);//return left,right,top,bottom
cv::Mat cutting(cv::Mat, int *);//return cuting image
float histogram(cv::Mat);
float asymmetry(cv::Mat, int *);// return degree of symmetry

int main(int argc, char* argv[]) {
    const int BUF_SIZE = 1280*720;
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


    if(::bind(serv_sock,(struct sockaddr*)&serv_addr,sizeof(serv_addr)) == -1)
        error_handling("bind() error");
    cout<<"bind()"<<endl;
    if(listen(serv_sock,30)==-1)
        error_handling("listen() error");
    cout<<"listen()"<<endl;
    for(int loop=0;loop<30;loop++){
        
        socklen_t clnt_addr_size = sizeof(clnt_addr);
        int clnt_sock = accept(serv_sock,(struct sockaddr*)&clnt_addr,&clnt_addr_size);
        if(clnt_sock==-1)
            error_handling("accept() error");
        cout<<"Connected"<<endl;

        int sum = 0;
        ofstream outFile("/Users/suhyeongcho/Desktop/Github/opencv/opencv/output.jpg");
        while(sum < 1280*720 ){
            str_len = read(clnt_sock,message,BUF_SIZE);
            //if(str_len <=0) break;
            sum += str_len;
            cout<<"str_len : "<<str_len<<endl;
            cout<<"sum : "<<sum<<endl;
            for(int i=0;i<str_len;i++)
                outFile<<message[i];
        }
        outFile.close();
        cout<<"sum : "<<sum<<endl;
    
        
        cv::Mat image = cv::imread("/Users/suhyeongcho/Desktop/Github/opencv/opencv/output.jpg", cv::IMREAD_COLOR);//원본 이미지
        cv::Mat original = cv::imread("/Users/suhyeongcho/Desktop/Github/opencv/opencv/output.jpg", cv::IMREAD_COLOR);//원본 이미지
        cv::Mat black;//기존 마스킹
        cv::Mat rough;//보다 낮은 값으로 마스킹
        cv::imshow("img", image);
        int row = image.rows;//세로
        int col = image.cols;//가로
        
        image = image(cv::Range(row/3, row/3*2), cv::Range(col/3, col/3*2));
        original = original(cv::Range(row/3, row/3*2), cv::Range(col/3, col/3*2));
        
        int * index;//black의 점의 좌 우 상 하 위치
        int * index2;//rough의 점의 좌 우 상 하 위치
        int resultA = -1, resultB = 0, resultC = -1;//판단 하려는 A B C의 %를 저장하려는 변수
        int cloudy = 55;//마스킹 하려는 범위를 결정하는 변수
        while (true) {//사진에서 점을 마스킹할 때 까지 반복
            black = masking(image, cloudy, cloudy);//점을 마스킹
            index = find(black);//마스킹된 점의 좌 우 상 하 위치 저장
            int sum = 0;
            for (int i = 0; i < 4; i++)
                sum += index[i];
            if (sum == 0) {//만약 좌표의 합이 0이면 마스킹이 안된 것임으로 cloudy를 10 낮춰서 다시 마스킹을 시도한다.
                cloudy -= 10;
            }
            else {
                break;
            }
        }
        fill(black);//마스킹된 점의내부에 빈 부분을 매워서 보정한다.
        
        rough = masking(image, cloudy - 30, cloudy);//black보다 낮은 값으로 마스킹
        index2 = find(rough);//마스킹된 점의 좌 우 상 하 위치 저장
        resultB = boundary(index, index2)*100;//두 좌표값의 차이로 B를 판단
        
        
        resultA = (int)asymmetry(black, index);//마스킹된 사진과 그 좌표로 A를 판단
        
        original = cutting(original, index);//사진을 마스킹된 사진으로 알게된 좌표로 잘라준다. 즉, 점만 남김
        resultC = (int)histogram(original);//잘린 점 사진을 이용해 C를 판단
        //cv::imshow("original", original);
        
        cout << "*********************" << endl;
        cout << "cloud : " << cloudy << endl;
        cout << "result A : " << resultA << "%" << endl;//100% 기준으로 대칭성
        cout << "result B : " << resultB << "%" << endl;//1이면 암
        cout << "result C : " << resultC << "%" << endl;//count/40*100
        cout << "*********************" << endl;

        char intStrA[10];
        char intStrB[10];
        char intStrC[10];
        
        sprintf(intStrA, "%d", resultA);
        sprintf(intStrB, "%d", resultB);
        sprintf(intStrC, "%d", resultC);

        
        strcat(intStrA,",");
        strcat(intStrA,intStrB);
        strcat(intStrA,",");
        strcat(intStrA,intStrC);
        
        cout<<intStrA<<endl;
        int result = write(clnt_sock,intStrA,strlen(intStrA));

        cout<<"close"<<endl;
        close(clnt_sock);

    }
    close(serv_sock);
    //cv::waitKey(0);
    
    return 0;
}


cv::Mat masking(cv::Mat image, int degree, int cloudy) {//60 50
    cv::Mat black;
    cv::Mat gray_image;
    medianBlur(image, image, 7);
    cv::cvtColor(image, gray_image, CV_BGR2GRAY); // 흑백영상으로 변환
    cv::adaptiveThreshold(gray_image, black, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 401, degree);
    if (degree == cloudy) {
        cv::imshow("Tight masking", black);
    }
    else {
        cv::imshow("Rough masking", black);
    }
    
    return black;//흑백으로 마스킹된 이미지 반환
}

int boundary(int * index1, int * index2) { // case 5 6 7 안됨
    int size1, size2;
    size1 = (index1[1] - index1[0])*(index1[3] - index1[2]);
    size2 = (index2[1] - index2[0])*(index2[3] - index2[2]);
    
    cout << "경계차이 비율 : " << (float)size2 / size1 * 100 << "\n";
    if (((float)size2 / size1 * 100) > 100 && ((float)size2 / size1 * 100) < 200) {
        cout << "경계선이 모호합니다" << "\n";
        return 1;
    }
    return 0;
}

void fill(cv::Mat black) {
    int row = black.rows;
    int col = black.cols;
    int read = -1;
    int start = 0;
    int end = 0;
    
    for (int i = 0; i < row; i++) {
        int x = col - 1;
        while (x >= 0) {
            read = black.at<uchar>(i, x);
            if (read == 0) {
                end = x;
                break;
            }
            x--;
        }
        x = 0;
        while (x < col) {
            read = black.at<uchar>(i, x);
            if (read == 0) {
                start = x;
                break;
            }
            x++;
        }
        for(int j = start ; j < end ; j++)
            black.at<uchar>(i, j) = 0;
        start = 0;
        end = 0;
    }
    
    for (int i = 0; i < col; i++) {
        int x = row - 1;
        while (x >= 0) {
            read = black.at<uchar>(x, i);
            if (read == 0) {
                end = x;
                break;
            }
            x--;
        }
        x = 0;
        while (x < row) {
            read = black.at<uchar>(x, i);
            if (read == 0) {
                start = x;
                break;
            }
            x++;
        }
        for (int j = start; j < end; j++)
            black.at<uchar>(j, i) = 0;
        start = 0;
        end = 0;
    }
}

int * find(cv::Mat black) {
    int row = black.rows;
    int col = black.cols;
    int left = 0, right = 0, top = 0, bottom = 0;
    int read = -1;
    bool flag = false;
    for (int i = 0; i < row; i++) {//find top
        for (int j = 0; j < col; j++) {
            read = black.at<uchar>(i, j);
            if (read == 0) {
                top = i;
                read = -1;
                flag = true;
                break;
            }
        }
        if (flag) {
            flag = false;
            break;
        }
    }
    for (int i = row - 1; i >= 0; i--) {//find bottom
        for (int j = 0; j < col; j++) {
            read = black.at<uchar>(i, j);
            if (read == 0) {
                bottom = i;
                read = -1;
                flag = true;
                break;
            }
        }
        if (flag) {
            flag = false;
            break;
        }
    }
    for (int i = 0; i < col; i++) {//find left
        for (int j = 0; j < row; j++) {
            read = black.at<uchar>(j, i);
            if (read == 0) {
                left = i;
                read = -1;
                flag = true;
                break;
            }
        }
        if (flag) {
            flag = false;
            break;
        }
    }
    for (int i = col - 1; i >= 0; i--) {//find right
        for (int j = 0; j < row; j++) {
            read = black.at<uchar>(j, i);
            if (read == 0) {
                right = i;
                read = -1;
                flag = true;
                break;
            }
        }
        if (flag) {
            flag = false;
            break;
        }
    }
    //cout << "left" << left << "right" << right << endl;
    //cout << "top" << top << "bottom" << bottom << endl;
    int * index = (int *)malloc(sizeof(int) * 4);
    index[0] = left;
    index[1] = right;
    index[2] = top;
    index[3] = bottom;
    return index;
}//884281
cv::Mat cutting(cv::Mat image, int * index) {//마스킹된 이미지를 기반으로 원본 이미지를 자름
    int left = index[0];
    int right = index[1];
    int top = index[2];
    int bottom = index[3];
    
    cv::Mat capture = image;
    if (!(left == 0 || right == 0 || top == 0 || bottom == 0))//예외 처리
        capture = image(cv::Range(top, bottom), cv::Range(left, right));
    
    return capture;//잘린 이미지 리턴
}

float asymmetry(cv::Mat image, int * index) {//날것의 마스킹
    int row = image.rows;
    int col = image.cols;
    int read = -1;
    int read2 = -1;
    int count = 0;
    
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            read = image.at<uchar>(i, j);
            if (read == 0) {
                count++;
            }
        }
    }
    double temp = sqrt((float)count / 3.141592);
    int r = (int)temp;
    
    int x = (index[1] - index[0]) / 2 + index[0];
    int y = (index[3] - index[2]) / 2 + index[2];
    cv::Mat background(row, col, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::circle(background, cv::Point(x,y), r+(count/600)+1, cv::Scalar(0,0,0),-1);
    cv::cvtColor(background, background, CV_BGR2GRAY);
    
    int sub = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            read = image.at<uchar>(i, j);
            read2 = background.at<uchar>(i, j);
            int temp = read - read2;
            if ((temp > 250) || (temp < -250)) {
                sub++;
            }
        }
    }
    int cir = 0;
    for(int i = 0 ; i < row ; i++){
        for(int j = 0 ; j < col ; j++){
            read = background.at<uchar>(i,j);
            if(read == 0)
                cir++;
        }
    }
    cout << "dot count : " << count << endl;
    cout<<"circle count : "<<cir<<endl;
    cout << "sub : " << sub << endl;
    cv::imshow("circle", background);
    sub = count - sub;
    /*cout << "count : " << count << endl;
     cout << "sub : " << sub << endl;
     cout << "answer : " << (float)sub / (float)count << endl;*/
    float result = 100 - (((float)sub / (float)count) * 100);
    if(result >= 100)
        return 100;
    else
        return result;
}

float histogram(cv::Mat capture) { // 30정도
    cv::Mat dst;
    cv::Mat bgr[3];
    cv::Mat hist; //Histogram 계산값 저장
    int channel[] = { 0,1,2 };
    int histSize = 255; //Histogram 가로값의 수
    int count = 0;
    float range[] = { 0,255.0 };
    const float * ranges = range;
    int hist_w = 512; int hist_h = 400;
    int number_bins = 255;
    int bin_w = cvRound((double)hist_w / number_bins);
    unsigned row2 = capture.rows; unsigned col2 = capture.cols; //자른 사진의 크기 저장
    
    cvtColor(capture, dst, CV_HSV2BGR); //Color 변경
    calcHist(&dst, 3, channel, cv::Mat(), hist, 1, &histSize, &ranges, true, false); //Histogram 계산
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    
    for (int i = 1; i < number_bins; i++) {    //Histogram 선 그리기
        line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))), cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))), cv::Scalar(0, 255, 0), 2, 8, 0);
    }
    
    for (int i = 0; i < histSize; i++) { //색의 다양성 검출
        //printf("%d번째 %f \n", i, hist.at<float>(i));
        if (hist.at<float>(i) > 229) {
            count++;
        }
    }
    
    printf("카운트 수 : %d\n", count);
    
    if (count > 30) {
        //printf("다양한 색조를 보입니다.\n");
    }
    else {
        //printf("다양한 색조를 보이지 않습니다.\n");
    }
    cv::namedWindow("Histogram", CV_WINDOW_AUTOSIZE);
    cv::imshow("HSV2BGR", dst);
    cv::imshow("Histogram", histImage);
    /*double p1 = 0.000002069;
     double p2 = -0.0002921;
     double p3 = 0.01324;
     double p4 = -0.1994;
     double p5 = 2.271;
     double p6 = - 1.959;    */
    double p1 = 0.000000577;
    double p2 = -0.00006435;
    double p3 = 0.00124;
    double p4 = 0.04832;
    double p5 = 0.7406;
    double p6 = -0.8064;
    double answer = p1 * pow(count, 5.0) + p2 * pow(count, 4.0) + p3 * pow(count, 3.0) + p4 * pow(count, 2.0) + p5 * pow(count, 1.0) + p6;
    if (answer >= 100) {
        answer = 100;
    }
    return answer;
}
