#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<cstdio>


using namespace cv;
using namespace std;
void diff(int *,int);
void sliceImage(int *, int,int&, int&);

int main(int argc, char* argv[]) {
    Mat dst1,dst2;
    
    //이미지 불러오기
    Mat image = imread("/Users/suhyeongcho/Desktop/opencv/src/cancer3.jpg", cv::IMREAD_COLOR);
    
    //이미지 -> 흰검 화 시키는 작업 + 저장
    cvtColor(image, dst1, CV_BGR2YCrCb);
    inRange(dst1, Scalar(0, 135, 80), Scalar(255, 171, 124), dst2);
    imwrite("/Users/suhyeongcho/Desktop/opencv/src/ycc_cancer3.jpg", dst2);
    
    //흰검 이미지 불러오기
    image = imread("/Users/suhyeongcho/Desktop/opencv/src/ycc_cancer3.jpg", cv::IMREAD_COLOR);
    
    
    int row = image.rows;//470 세로
    int col = image.cols;//624 가로
    
    int width[624] = { 0 };
    int width2[470] = {0};
    float width_f[624] = {0.0};
    float width2_f[470] = {0.0};
    int red = 0,green = 0,blue = 0;
    
    //자르기자르기자르기
    int col_start = 0, col_end = 0;
    int row_start = 0, row_end = 0;
    //세로축 자르기 위한 rgb값 평균 구하는 과정
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            red = image.at<Vec3b>(i, j)[2];
            green = image.at<Vec3b>(i, j)[1];
            blue = image.at<Vec3b>(i, j)[0];
            width[j] += (red + green + blue);
        }
    }
    
    for (int i = 0; i < col; i++) {
        width_f[i] = (float)width[i] / (3 * row);
        width[i] = (int)width_f[i];
    }
    //세로축 자르기
    sliceImage(width,col,col_start,col_end);
    
    //가로축 자르기 위한 rgb값 평균 구하는 과정
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            red = image.at<Vec3b>(i, j)[2];
            green = image.at<Vec3b>(i, j)[1];
            blue = image.at<Vec3b>(i, j)[0];
            width2[i] += (red + green + blue);
        }
    }
    
    for (int i = 0; i < row; i++) {
        width2_f[i] = (float)width2[i] / (3 * col);
        width2[i] = (int)width2_f[i];
    }
    //가로축 자르기
    sliceImage(width2,row,row_start,row_end);
    
    
    cout<<"row : "<<row_start<<','<<row_end<<endl;
    cout<<"col : "<<col_start<<','<<col_end<<endl;
    
    //이미지를 자르기 -> cut에 저장
    Mat cut;
    cut = image(Range(row_start,row_end),Range(col_start,col_end));
    
    //이미지 자른거 사분할하기
    //1 0
    //2 3 순서
    Mat quarter[4];
    Mat flip_quarter[4];
    int cut_row = cut.rows;
    int cut_col = cut.cols;
    quarter[0] = cut(Range(0,cut_row/2),Range(cut_col/2,cut_col));
    quarter[1] = cut(Range(0,cut_row/2),Range(0,cut_col/2));
    quarter[2] = cut(Range(cut_row/2,cut_row),Range(0,cut_col/2));
    quarter[3] = cut(Range(cut_row/2,cut_row),Range(cut_col/2,cut_col));
    
    //사분할 이미지 뒤집기
    flip_quarter[0] = quarter[0];
    flip(quarter[1], flip_quarter[1], 1);
    flip(quarter[2], flip_quarter[2], -1);
    flip(quarter[3], flip_quarter[3], 0);
    
    //이미지 픽셀 값 차이 나는지 확인해서 확률 구하기
    int sum1 = 0,sum2 = 0;
    int size = 0;
    for(int i=0;i<flip_quarter[0].rows;i++){
        for(int j=0;j<flip_quarter[0].cols;j++){
            size++;
            int num1 = flip_quarter[0].at<uchar>(i,j)-flip_quarter[2].at<uchar>(i,j);
            int num2 = flip_quarter[1].at<uchar>(i,j)-flip_quarter[3].at<uchar>(i,j);
            if(num1) sum1++;
            if(num2) sum2++;
        }
    }
    
    cout<<1-(double)sum1/size<<endl;
    cout<<1-(double)sum2/size<<endl;

    
//    namedWindow("a");
//    namedWindow("b");
//    namedWindow("c");
//    namedWindow("d");
//    namedWindow("e");
//    namedWindow("f");
//
//    imshow("a",image);
//    imshow("b",cut);
//    imshow("c", flip_quarter[0]);
//    imshow("d", flip_quarter[1]);
//    imshow("e", flip_quarter[2]);
//    imshow("f", flip_quarter[3]);
//    waitKey();
}

//문돌이가 구현한 자르기 함수
void sliceImage(int *width,int size,int& start,int& end){
    
    diff(width, size);
    
    //여기서 대충 계산 되었다고 치고
    int long_under = 0, long_over = 0;
    int cur_under = 0, cur_over = 0;
    
    int cur_start = 0, cur_end = 0;
    
    int mid = size/2;
    
    for (int i = 0; i < mid; i++) {
        if (width[i] < 0) {//값이 감소추세일 때
            cur_under++;
            if (cur_under == 1) {
                cur_start = i;
            }
        }
        else {//증가추세일 때
            if (cur_under > long_under) {//최근 측정 길이가 가장 길다면
                start = cur_start;
                long_under = cur_under;
            }
            cur_under = 0;
            cur_start = 0;
        }
        if(!start) start = cur_start;
    }
    
    for (int i = size-2; i > mid; i--) {
        if (width[i] < 0) {//값이 감소 추세일 때
            cur_over++;
            if (cur_over == 1) {
                cur_end = i;
            }
        }
        else {//증가추세일 때
            if (cur_over > long_over) {//최근 측정 길이가 가장 길다면
                end = cur_end;
                long_over = cur_over;
            }
            cur_over = 0;
            cur_end = 0;
        }
        if(!end) end = cur_end;
    }
}

//다음꺼 뺴기 지금꺼 함수// 양수면 지금꺼가 더 어둡다 // 음수면 다음꺼가 더 어둡다
void diff(int * width, int col) {
    for (int i = 0; i < col-1; i++) {
        width[i] = width[i + 1] - width[i];
    }
    width[col - 1] = 0;
}


/*
 int main() {
 
 Mat image;
 Mat dst1,dst2,dst3;
 //Mat cut[4];
 //Mat dst[4];
 //Mat bgr[4][3];
 //Mat hist[4];
 //int channel[] = {0,1};
 //int histSize[] = {256};
 //float range[]={0,256};
 //const float * ranges[] = {range};
 
 image = imread("/Users/suhyeongcho/Desktop/opencv/src/cancer3.jpg", CV_LOAD_IMAGE_COLOR);
 //unsigned row = image.rows; unsigned col = image.cols;
 cvtColor(image, dst1, CV_BGR2YCrCb);
 //inRange(dst1, Scalar(0, 133, 77), Scalar(255, 173,127), dst2);
 //inRange(dst1, Scalar(0, 135, 81), Scalar(255, 171, 123), dst2);
 inRange(dst1, Scalar(0, 135, 80), Scalar(255, 171, 124), dst3);
 
 // imwrite("/Users/suhyeongcho/Desktop/opencv/src/ycc_cancer3.jpg", dst3);
 namedWindow("image");
 //namedWindow("dst1");
 //namedWindow("dst2");
 namedWindow("dst3");
 
 imshow("image", image);
 //imshow("dst1", dst1);
 //imshow("dst2", dst2);
 imshow("dst3",dst3);
 
 waitKey();
 
 
 
 
 //    flip(image, cut[1], 1);//좌우반전
 //    flip(image, cut[2], -1);//다 반전
 //    flip(image, cut[3], 0);//상하반전
 //
 //
 //    cut[0] = image;
 //    for(int i=0;i<4;i++) cut[i] = cut[i](Range(0,row/2),Range(0,col/2));
 //
 //    row = row/2;
 //    col = col/2;
 //
 //    cout<<row<<" "<<col<<endl;
 //    int red[4][row][col],blue[4][row][col],green[4][row][col];
 //    for(int i=0;i<4;i++)
 //        for(int j=0;j<row;j++)
 //            for(int k=0;k<col;k++){
 //                red[i][j][k] = cut[i].at<Vec3b>(j,k)[2];
 //                green[i][j][k] = cut[i].at<Vec3b>(j,k)[1];
 //                blue[i][j][k] = cut[i].at<Vec3b>(j,k)[0];
 //            }
 //
 //    for(int i=0;i<row;i++){
 //        for(int j=0;j<col;j++){
 //            int a = red[0][i][j]-red[2][i][j];
 //            a =(a>=0)?a:-a;
 //            cout<<a<<',';
 //        }
 //        cout<<endl<<endl;
 //    }
 //
 //    Mat diff;
 //    subtract(cut[0], cut[2], diff);
 //    cout<<"cut0 : "<<cut[0].rows<<","<<cut[0].cols<<endl;
 //    cout<<"cut2 : "<<cut[2].rows<<","<<cut[2].cols<<endl;
 //
 //
 //    for(int i=0;i<row;i++)
 //        for(int j=0;j<col;j++){
 //            int a = red[0][i][j]-red[2][i][j];
 //            a =(a>=0)?a:-a;
 //            diff.at<uchar>(i,j) = a;
 //        }
 //
 //    namedWindow("a");
 //    imshow("a",diff);
 //    waitKey();
 //
 //    for(int i=0;i<4;i++) cvtColor(cut[i], dst[i], CV_HSV2BGR);
 //
 //    for(int i=0;i<4;i++) calcHist(&dst[i], 1, channel, Mat(), hist[i], 1, histSize, ranges,true,false);
 //
 //
 //    double result[4][4];
 //
 //    for(int i=0;i<4;i++)
 //        for(int j=0;j<4;j++)
 //            result[i][j] = compareHist(hist[i], hist[j], 0);
 //
 //    for(int i=0;i<4;i++){
 //        for(int j=0;j<4;j++){
 //            printf("%.6f ",result[i][j]);
 //        }
 //        cout<<endl;
 //    }
 
 return 0;
 
 }*/


