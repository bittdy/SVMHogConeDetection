#define NEED_TRAIN 0

#include "cv.h"  
#include "highgui.h"  
#include <ml.h>  
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>  
#include <sstream>
using namespace cv;  
using namespace std;  
  
  
int main(int argc, char** argv)    
{    
	int k=0;
	int j=0;
	int imageNumber = 10000;
	stringstream imageStream;
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
    IplImage* trainImg=cvCreateImage(cvSize(64,64),8,1);//��Ҫ������ͼƬ������Ĭ���趨ͼƬ��64*64��С���������涨����1764�����Ҫ����ͼƬ��С����������debug�鿴һ��descriptors�Ƕ��٣�Ȼ���趨��������  
  
	//��ʼ��HOG����
    for( string::size_type i = 0; i != img_path.size(); i++ )  
    {  
            src=cvLoadImage(img_path[i].c_str(),0);  
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

	IplImage* test = cvLoadImage("test.jpg",0);
	IplImage* test_copy = cvCreateImage(cvSize(test->width,test->height),8,1);
	IplImage* test_copy1 = cvCreateImage(cvSize(test->width,test->height),8,1);
	cvCopy(test,test_copy,NULL);
	cvCopy(test,test_copy1,NULL);
	for(k = 0; k < test->width-64; k=k+32)
	{
		for(j=0; j < test->height-64; j=j+32)
		{
			cvRectangle(test_copy1,cvPoint(k,j),cvPoint(k+63,j+63),Scalar(0,0,0),1,8,0);
			cvNamedWindow("imgNow",1);
			cvShowImage("imgNow",test_copy1);
			//cvWaitKey(0);
			cvSetImageROI(test, cvRect(k, j, 64, 64));
			IplImage *img = cvCreateImage(cvGetSize(test), test->depth,test->nChannels);
			cvCopy(test, img, NULL);
			cvNamedWindow("imgDetecting",1);
			cvShowImage("imgDetecting",img);
			//cvWaitKey(0);
			cvResetImageROI(test);
			HOGDescriptor *hog1=new HOGDescriptor(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);  //������˼���ο�����1,2     
			vector<float>descriptors1;//�������     
			hog1->compute(img, descriptors1,Size(1,1), Size(0,0)); //���ü��㺯����ʼ����     
			cout<<"HOG dims: "<<descriptors1.size()<<endl;  
			CvMat* SVMtrainMat=cvCreateMat(1,descriptors1.size(),CV_32FC1);  
			n=0;  
			for(vector<float>::iterator iter1=descriptors1.begin();iter1!=descriptors1.end();iter1++)  
			{  
				cvmSet(SVMtrainMat,0,n,*iter1);  
				n++;  
			}    
			int ret = svm.predict(SVMtrainMat);//��ȡ���ռ���������predict���÷��� OpenCV���ĵ� 
			if(ret == 1)
			{
				cvRectangle(test_copy,cvPoint(k,j),cvPoint(k+63,j+63),Scalar(0,255,255),1,8,0);//�������Ϊ׮Ͱ��roi������ȷ��ѵ��������,����֮������
				imageNumber++;
				imageStream<<imageNumber;
				string imageName = "F:\\vs2010_projects\\OpencvSvm\\OpencvSvm\\hardexample\\"+imageStream.str()+".jpg";
				char *saveImageName = (char *)imageName.data();
				cvSaveImage(saveImageName,img);
				imageStream.str("");
				cvNamedWindow("img",1);
				cvShowImage("img",img);
 				//cvWaitKey(0);
			}
			if(j+63>test->height-64)   //����ǰ���Ե����64����ͼ���ʱ˵���Ѿ��м����ɣ���ֹѭ��
				break;
		}
		if(k+63>test->width-64)   //����ǰ���Ե����64����ͼ���ʱ˵���Ѿ��м����ɣ���ֹѭ��
			break;
	}
cvNamedWindow("finalresult",1);
cvShowImage("finalresult",test_copy);
cvWaitKey(0);
  
//cvReleaseImage( &src);  
//cvReleaseImage( &sampleImg );  
//cvReleaseImage( &tst );  
//cvReleaseImage( &tst_tmp );  
cvReleaseMat( &data_mat );  
cvReleaseMat( &res_mat );  
  
return 0;  
}  