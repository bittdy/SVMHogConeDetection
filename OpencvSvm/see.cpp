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
    ifstream svm_data( "data.txt" );//���ȣ������һ���ļ��б���ѵ������ͼƬ��·����д�����txt�ļ��У�ʹ��bat�������ļ����Եõ����txt�ļ�   
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
    data_mat = cvCreateMat( nImgNum, 365904, CV_32FC1 );  //����ڶ���������������������������descriptors�Ĵ�С�����ģ�������descriptors.size()�õ����Ҷ��ڲ�ͬ��С������ѵ��ͼƬ�����ֵ�ǲ�ͬ��
    cvSetZero( data_mat );  
    //���;���,�洢ÿ�����������ͱ�־  
    res_mat = cvCreateMat( nImgNum, 1, CV_32FC1 );  
    cvSetZero( res_mat );  
  
    IplImage* src; 
string path;
    IplImage* trainImg=cvCreateImage(cvSize(36,136),8,3);//��Ҫ������ͼƬ������Ĭ���趨ͼƬ��64*64��С���������涨����1764�����Ҫ����ͼƬ��С����������debug�鿴һ��descriptors�Ƕ��٣�Ȼ���趨��������  
  
 vector<float>descriptors;//�������  
//��ʼ��HOG����
    HOGDescriptor *hog=new HOGDescriptor(cvSize(16,16),cvSize(8,8),cvSize(8,8),cvSize(4,4),9);  //������˼���ο�����1,2     

    for( string::size_type i = 0; i != img_path.size(); i++ )  
    {  
            path=img_path[i];
    src=cvLoadImage(path.c_str(),1);  
            if( src == NULL )  
            {  
                cout<<" can not load the image: "<<img_path[i].c_str()<<endl;  
                continue;  
            }  
  
            cout<<" processing "<<img_path[i].c_str()<<endl;  
                 
            cvResize(src,trainImg);   //��ȡͼƬ     
           
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
//{
//	vector<float>().swap(descriptors); //�ͷ�vector�ڴ�
//}
            cout<<" end processing "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl; 
cvReleaseImage( &src );
src=NULL;

    } 
cout<<endl;
    cout<<"��1��Hog��ȡ������������ʼѵ��������"<<endl; 
cout<<endl;
               
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
           
    svm.train( data_mat, res_mat, NULL, NULL, param );//ѵ����    
    svm.save( "SVM.xml" );
  
cout<<endl;
cout<<"��2��ѵ���������Ա���ѵ�����ݣ�SVM.xml"<<endl; 
cout<<endl;

    //�������  
    IplImage *test;  
    vector<string> img_tst_path;  
    ifstream img_tst( "test.txt" );//ͬ����ѵ������������Ҳ��һ���ģ�ֻ��������Ҫ��עͼƬ������һ����
    while( img_tst )  
    {  
        if( getline( img_tst, buf ) )  
        {  
            img_tst_path.push_back( buf );  
        }  
    }  
    img_tst.close();  
  
  
cout<<endl;
cout<<"��3����ʼ����������"<<endl; 
cout<<endl;

    CvMat *test_hog = cvCreateMat( 1, 365904, CV_32FC1 );//ע�������1764��ͬ����һ�� 
CvMat* SVMtrainMat;
    char line[512];  
    ofstream predict_txt( "preditcion.txt" );//��Ԥ�����洢������ı���  
CvSVM svm_hog;
svm_hog.load("SVM.xml");

    for( string::size_type j = 0; j != img_tst_path.size(); j++ )//���α������еĴ����ͼƬ  
    {  
        path=img_tst_path[j];
test = cvLoadImage(path.c_str(), 1);  
        if( test == NULL )  
        {  
             cout<<" can not load the image: "<<img_tst_path[j].c_str()<<endl;  
               continue;  
         }  
          
        cvZero(trainImg);  
        cvResize(test,trainImg);   //��ȡͼƬ     
       //vector<float>descriptors;//�������     
        hog->compute(trainImg, descriptors,Size(1,1), Size(0,0)); //���ü��㺯����ʼ����     
       
cout<<"HOG dims: "<<descriptors.size()<<endl;  
       
if (j==0)
{
SVMtrainMat=cvCreateMat(1,descriptors.size(),CV_32FC1); 
}	 
        n=0;  
        for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)  
            {  
                cvmSet(SVMtrainMat,0,n,*iter);  
                n++;  
            }  
  
        int ret = svm_hog.predict(SVMtrainMat);//��ȡ���ռ���������predict���÷��� OpenCV���ĵ� 
std::sprintf( line, "%s %d\r\n", img_tst_path[j].c_str(), ret ); 

/*{
vector<float>().swap(descriptors);
}*/
        predict_txt<<line; 
cvReleaseImage( &test );
test=NULL;

    }  
    predict_txt.close();  
  


cvReleaseImage( &trainImg);

cvReleaseMat( &test_hog );  
cvReleaseMat( &data_mat );  
cvReleaseMat( &res_mat );  
    
return 0;  
}  