
#include "cv.h"  
#include "highgui.h"  
#include <ml.h>  
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>  
using namespace cv;  
using namespace std;  
  
  
int main(int argc, char** argv)    
{    
    vector<string> img_path;//�����ļ������� 
    vector<int> img_catg;  
    int nLine = 0;  
    string buf;  
    ifstream svm_data( "trainSample.txt" );//���ȣ������һ���ļ��б���ѵ������ͼƬ��·����д�����txt�ļ��У�ʹ��bat�������ļ����Եõ����txt�ļ�   
    unsigned long n;  
  
    while( svm_data )//��ѵ�������ļ����ζ�ȡ����  
    {  
        if( getline( svm_data, buf ) )  
        {  
            nLine ++;  
            if( nLine % 2 == 0 )//����ķ���Ƚ�����˼�����ó��������SVM_DATA.txt�ı���Ӧ����һ�����ļ�·����������һ�о��Ǹ�ͼƬ����𣬿�������Ϊ0����1����Ȼ���Ҳ����ν 
            {  
                 img_catg.push_back( atoi( buf.c_str() ) );//atoi���ַ���ת�������ͣ���־��0,1����ע����������Ҫ��������𣬷�������  
            }  
            else  
            {  
                img_path.push_back( buf );//ͼ��·��  
            }  
        }  
    }  
    svm_data.close();//�ر��ļ�  
  
    CvMat *data_mat, *res_mat;  
    int nImgNum = nLine / 2; //������������ ����Ϊ��ÿ��һ�в���ͼƬ·��������Ҫ����2 
    ////��������nImgNum�������������������� WIDTH * HEIGHT������������������ͼ���С  
    data_mat = cvCreateMat( nImgNum, 1764, CV_32FC1 );  //����ڶ���������������������������descriptors�Ĵ�С�����ģ�������descriptors.size()�õ����Ҷ��ڲ�ͬ��С������ѵ��ͼƬ�����ֵ�ǲ�ͬ��
    cvSetZero( data_mat );  
    //���;���,�洢ÿ�����������ͱ�־  
    res_mat = cvCreateMat( nImgNum, 1, CV_32FC1 );  
    cvSetZero( res_mat );  
  
    IplImage* src;  
    IplImage* trainImg=cvCreateImage(cvSize(64,64),8,3);//��Ҫ������ͼƬ������Ĭ���趨ͼƬ��64*64��С���������涨����1764�����Ҫ����ͼƬ��С����������debug�鿴һ��descriptors�Ƕ��٣�Ȼ���趨��������  
  
	//��ʼ��HOG����
    for( string::size_type i = 0; i != img_path.size(); i++ )  
    {  
            src=cvLoadImage(img_path[i].c_str(),1);  
            if( src == NULL )  
            {  
                cout<<" can not load the image: "<<img_path[i].c_str()<<endl;  
                continue;  
            }  
  
            cout<<" processing "<<img_path[i].c_str()<<endl;  
                 
            cvResize(src,trainImg);   //��ȡͼƬ     
            HOGDescriptor *hog=new HOGDescriptor(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);  //������˼���ο�����1,2     
            vector<float>descriptors;//�������     
            hog->compute(trainImg, descriptors,Size(1,1), Size(0,0)); //���ü��㺯����ʼ����     
            cout<<"HOG dims: "<<descriptors.size()<<endl;  
            //CvMat* SVMtrainMat=cvCreateMat(descriptors.size(),1,CV_32FC1);  
            n=0;  
            for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)  
            {  
                cvmSet(data_mat,i,n,*iter);//��HOG�洢����  
                n++;  
            }  
                //cout<<SVMtrainMat->rows<<endl;  
            cvmSet( res_mat, i, 0, img_catg[i] );  
            cout<<" end processing "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;  
    }  
      
               
    CvSVM svm;//�½�һ��SVM    
    CvSVMParams param;//�����ǲ���
    CvTermCriteria criteria;    
    criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );    
    param = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );    
/*    
    SVM���ࣺCvSVM::C_SVC    
    Kernel�����ࣺCvSVM::RBF    
    degree��10.0���˴β�ʹ�ã�    
    gamma��8.0    
    coef0��1.0���˴β�ʹ�ã�    
    C��10.0    
    nu��0.5���˴β�ʹ�ã�    
    p��0.1���˴β�ʹ�ã�    
    Ȼ���ѵ���������滯����������CvMat�͵������    
                                                        */       
    //����������(5)SVMѧϰ�������������         
    svm.train( data_mat, res_mat, NULL, NULL, param );//ѵ����    
    //�������ѵ�����ݺ�ȷ����ѧϰ����,����SVMѧϰ�����     
    svm.save( "SVM_DATA.xml" );
  
    //�������  
    IplImage *test;  
    vector<string> img_tst_path;  
    ifstream img_tst( "testSample.txt" );//ͬ����ѵ������������Ҳ��һ���ģ�ֻ��������Ҫ��עͼƬ������һ����
    while( img_tst )  
    {  
        if( getline( img_tst, buf ) )  
        {  
            img_tst_path.push_back( buf );  
        }  
    }  
    img_tst.close();  
  
  
  
    CvMat *test_hog = cvCreateMat( 1, 1764, CV_32FC1 );//ע�������1764��ͬ����һ��  
    char line[512];  
    ofstream predict_txt( "SVM_PREDICT.txt" );//��Ԥ�����洢������ı���  
    for( string::size_type j = 0; j != img_tst_path.size(); j++ )//���α������еĴ����ͼƬ  
    {  
        test = cvLoadImage( img_tst_path[j].c_str(), 1);  
        if( test == NULL )  
        {  
             cout<<" can not load the image: "<<img_tst_path[j].c_str()<<endl;  
               continue;  
         }  
          
        cvZero(trainImg);  
        cvResize(test,trainImg);   //��ȡͼƬ     
        HOGDescriptor *hog=new HOGDescriptor(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);  //������˼���ο�����1,2     
        vector<float>descriptors;//�������     
        hog->compute(trainImg, descriptors,Size(1,1), Size(0,0)); //���ü��㺯����ʼ����     
        cout<<"HOG dims: "<<descriptors.size()<<endl;  
        CvMat* SVMtrainMat=cvCreateMat(1,descriptors.size(),CV_32FC1);  
        n=0;  
        for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)  
            {  
                cvmSet(SVMtrainMat,0,n,*iter);  
                n++;  
            }  
  
        int ret = svm.predict(SVMtrainMat);//��ȡ���ռ���������predict���÷��� OpenCV���ĵ� 
		std::sprintf( line, "%s %d\r\n", img_tst_path[j].c_str(), ret );  
        predict_txt<<line;  
    }  
    predict_txt.close();  
  
//cvReleaseImage( &src);  
//cvReleaseImage( &sampleImg );  
//cvReleaseImage( &tst );  
//cvReleaseImage( &tst_tmp );  
cvReleaseMat( &data_mat );  
cvReleaseMat( &res_mat );  
  
return 0;  
}  