/*******************************
		ͼ��ο������
		��ͨ·׶ʶ��
		��������СȻ�����������ӫ
		2017.10.22
 *******************************/
#define NEED_TRAIN 1
#define NEED_TRAIN_HARDEX 0
#include "cv.h"  
#include "highgui.h"  
#include <ml.h>  
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>  
#include <sstream>
#include <vector>
#include <Eigen\Dense>
#include <Eigen\core>
using namespace cv;  
using namespace std;  

int redLow = 0;
int redHigh = 15;
int yellowLow = 18;
int yellowHigh = 40;
int blueLow = 101;
int blueHigh = 120;
  
int areaCount(CvPoint P1,CvPoint P2)
{
	return ((P1.x - P2.x)*(P1.y - P2.y));
}
int isOverLap(CvPoint P1,CvPoint P2,CvPoint PN1,CvPoint PN2)
{
	//return 0 when there is overlap otherwise return 1
	if(P1.x>PN2.x){return 1;}
	if(PN1.x>P2.x){return 1;}
	if(P2.y<PN1.y){return 1;}
	if(PN2.y<P1.y){return 1;}
	return 0;
}
Eigen::Vector3f ColorCluster(cv::InputArray inImage, cv::OutputArray outImage)
{
	// kmeans����Ԥ����
    Mat Image = inImage.getMat();
    Mat samples(Image.cols * Image.rows, 1, CV_32FC3);
    Mat labels(Image.cols * Image.rows, 1, CV_32SC1);
    uchar* p;
    int i, j, k = 0;
    for (i = 0; i < Image.rows; i++)
    {
        p = Image.ptr<uchar>(i);
        for (j = 0; j< Image.cols; j++)
        {
            samples.at<Vec3f>(k, 0)[0] = float(p[j * 3]);
            samples.at<Vec3f>(k, 0)[1] = float(p[j * 3 + 1]);
            samples.at<Vec3f>(k, 0)[2] = float(p[j * 3 + 2]);
            k++;
        }
    }
    int clusterCount = 2;
    Mat centers(clusterCount, 1, samples.type());
    kmeans(samples, clusterCount, labels,
        TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 5, 1.0),
        clusterCount, KMEANS_PP_CENTERS, centers);

	// ��þ�������������۳ɵ���������
    Mat imgClustered(Image.rows, Image.cols, CV_8UC1);
    float step = 255 / (clusterCount - 1);
    k = 0;
    int class0_sum=0, class1_sum=0;
    for (i = 0; i < imgClustered.rows; i++)
    {
        p = imgClustered.ptr<uchar>(i);
        for (j = 0; j< imgClustered.cols; j++)
        {
            int tt = labels.at<int>(k, 0);
            if(tt == 0)
                class0_sum++;
            else
                class1_sum++;
            p[j] = 255 - tt*step;
            k++;
        }
    }
    //cout << "class0_sum = " << class0_sum << ", class1_sum = " << class1_sum << endl;

    //cout << "centers = \n" << centers << endl;
	
	// �۳ɵ�����ľ�������
    Eigen::Vector3f center1, center2;
    center1[0] = centers.at<float>(0,0);
    center1[1] = centers.at<float>(0,1);
    center1[2] = centers.at<float>(0,2);
    
    center2[0] = centers.at<float>(1,0);
    center2[1] = centers.at<float>(1,1);
    center2[2] = centers.at<float>(1,2);

    //cout << "center1 = \n" << center1 << endl;
    //cout << "center2 = \n" << center2 << endl;

    Eigen::Vector3f colorVec;
    int id;
	int case1,case2;

	//if(center1[0] == 0 && center1[1] == 0 && center1[2] == 0)
     //   return -1; // invalid
	
    if(center1[0]>=redLow && center1[0]<=redHigh)  // Red
        case1 = 2;
    else if(center1[0]>yellowLow && center1[0]<=yellowHigh)  // Yellow
        case1 = 1;
    else if(center1[0]>=blueLow && center1[0]<=blueHigh)  // Blue
        case1 = 0;
    else
        case1 = -3;  //unknown

	//if(center2[0] == 0 && center2[1] == 0 && center2[2] == 0)
     //   return -1; // invalid
	
    if(center2[0]>=redLow && center2[0]<=redHigh)  // Red
        case2 = 2;
    else if(center2[0]>yellowLow && center2[0]<=yellowHigh)  // Yellow
        case2 = 1;
    else if(center2[0]>=blueLow && center2[0]<=blueHigh)  // Blue
        case2 = 0;
    else
        case2 = -3;  //unknown
	// �ж�����
	// 1��С�Ĳ��ֵ�����Ķ���С�ڴ�Ĳ��֣��ɵ�����
	// 2��С�Ĳ��ֵ�Sͨ������35���ɵ�����
   if(-3!=case1 && -3!=case2)  //����������������Ϊunknow���ҳ�����
   {
			if(1 * class0_sum < class1_sum && center1[1] > 35)
		{
			colorVec = center1;
			id = 0;
		}
		else if(2 * class1_sum < 2 * class0_sum && center2[1] > 35)
		{
			colorVec = center2;
			id = 1;
		}
		else //����λ invalid ���ԣ���ɫ����ȫ����Ϊ0����ЩǷ�ף����Լ�һ�����ԣ��Ա����ɫ�������أ�������Σ�
		{
			Eigen::Vector3f invalid;
			invalid.fill(0);
			colorVec = invalid;
			id = 2;
		}
   }
   else if(-3 == case2)
   {
		colorVec = center1;
		id = 0;
   }
   else 
   {
		colorVec = center2;
		id = 1;
   }

    imgClustered.copyTo(outImage);
    //cout << "id = " << id <<", colorVec = " << colorVec.transpose() << endl;
    return colorVec;
}

//Ϊ���Գ��Գ���Ķ���С���ú����Ա�����ʵ�ʿ���ֱ��������ĺ���ֱ�Ӽ��
int ExtractColor(const Eigen::Vector3f color) // ֱ���˹�ȷ�������������ɫ���䣬��ͨ��Hͨ���жϡ��������ɵ�����
{
    cout << "color = " << color.transpose() << endl;

    if(color[0] == 0 && color[1] == 0 && color[2] == 0)
        return -1; // invalid
	
    if(color[0]>=redLow && color[0]<=redHigh)  // Red
        return 2;
    else if(color[0]>yellowLow && color[0]<=yellowHigh)  // Yellow
        return 1;
    else if(color[0]>=blueLow && color[0]<=blueHigh)  // Blue
        return 0;
    else
        return -3;  //unknown

}
  
int main(int argc, char** argv)    
{    
	int k=0;
	int j=0;
	int imageNumber = 10000;
	stringstream imageStream;
    vector<string> img_path;//�����ļ������� 
    vector<int> img_catg;  
	vector<CvPoint> detect_window_up;
	vector<CvPoint> detect_window_down;
    int nLine = 0;  
    string buf;  
    ifstream svm_data( "trainSample.txt" );//���ȣ������һ���ļ��б���ѵ������ͼƬ��·����д�����txt�ļ��У�ʹ��bat�������ļ����Եõ����txt�ļ�   
    unsigned long n;  
  
    while( svm_data )//��ѵ�������ļ����ζ�ȡ����  
    {  
        if( getline( svm_data, buf ) )  
        {  
            nLine ++;  
            if( nLine % 2 == 0 )//SVM_DATA.txt�ı���Ӧ����һ�����ļ�·����������һ�о��Ǹ�ͼƬ����𣬿�������Ϊ0����1����Ȼ���Ҳ����ν 
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
    svm.train( data_mat, res_mat, NULL, NULL, param );//ѵ��    
    //�������ѵ�����ݺ�ȷ����ѧϰ����,����SVMѧϰ�����     
    svm.save( "SVM_DATA.xml" );

	IplImage* originSrc = cvLoadImage("test1.jpg",1);
	IplImage* test = cvCreateImage(cvSize(originSrc->width,originSrc->height),8,1);
	cvCvtColor(originSrc , test , CV_BGR2GRAY);//Դͼ���Ŀ��ͼ�񶼱�����IPL_DEPTH_32F��
	
	IplImage* test_copy = cvCreateImage(cvSize(test->width,test->height),8,1);
	IplImage* test_copy1 = cvCreateImage(cvSize(test->width,test->height),8,1);
	cvCopy(test,test_copy,NULL);
	cvCopy(test,test_copy1,NULL);
	CvPoint P1_1,P1_2;
	for(k = 0; k < test->width-64; k=k+16)
	{
		for(j=0; j < test->height-64; j=j+16)
		{
			cvRectangle(test_copy1,cvPoint(k,j),cvPoint(k+63,j+63),Scalar(0,0,0),1,8,0);
			//cvNamedWindow("imgNow",1);
			//cvShowImage("imgNow",test_copy1);
			//cvWaitKey(0);
			cvSetImageROI(test, cvRect(k, j, 64, 64));
			IplImage *img = cvCreateImage(cvGetSize(test), test->depth,test->nChannels);
			cvCopy(test, img, NULL);
			//cvNamedWindow("imgDetecting",1);
			//cvShowImage("imgDetecting",img);
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
				//imageNumber++;
				//imageStream<<imageNumber;
				//string imageName = "F:\\vs2010_projects\\OpencvSvm\\OpencvSvm\\hardexample\\"+imageStream.str()+".jpg";
				//char *saveImageName = (char *)imageName.data();
				//cvSaveImage(saveImageName,img);
				//imageStream.str("");
				//cvNamedWindow("img",1);
				//cvShowImage("img",img);
 				//cvWaitKey(0);
				if(0==detect_window_up.size())
				{
					detect_window_up.push_back(cvPoint(k,j));
					detect_window_down.push_back(cvPoint(k+63,j+63));
					continue;
				}
				int vector_size = detect_window_up.size();
				for(int m=0;m<vector_size;m++)
				{						
					P1_1.x = max(k,detect_window_up.at(m).x);
					P1_1.y = max(j,detect_window_up.at(m).y);
					P1_2.x = min(k+63,detect_window_down.at(m).x);
					P1_2.y = min(j+63,detect_window_down.at(m).y);
					if((0==isOverLap(cvPoint(k,j),cvPoint(k+63,j+63),detect_window_up.at(m),detect_window_down.at(m)))&&(double(areaCount(P1_1,P1_2))/(64*64)>=0.4))//˵���͵�m�������ص����ص�����ﵽһ�����
					{
						CvPoint cvPointBuffer;
						cvPointBuffer.x = min(k,detect_window_up.at(m).x);
						cvPointBuffer.y = min(j,detect_window_up.at(m).y);
						detect_window_up.at(m).x = cvPointBuffer.x;
						detect_window_up.at(m).y = cvPointBuffer.y;
						cvPointBuffer.x = max(k+63,detect_window_down.at(m).x);
						cvPointBuffer.y = max(j+63,detect_window_down.at(m).y);
						detect_window_down.at(m).x = cvPointBuffer.x ;
						detect_window_down.at(m).y = cvPointBuffer.y;
						break;
					}
					if(m==vector_size-1)
					{
						detect_window_up.push_back(cvPoint(k,j));
						detect_window_down.push_back(cvPoint(k+63,j+63));
					}
				}
				
			}
			if(j+63>test->height)   //����ǰ���Ե����64����ͼ���ʱ˵���Ѿ��м����ɣ���ֹѭ��
				break;
		}
		if(k+63>test->width)   //����ǰ���Ե����64����ͼ���ʱ˵���Ѿ��м����ɣ���ֹѭ��
			break;
	}
	//cvNamedWindow("finalresult",1);
	//cvShowImage("finalresult",test_copy);
	//cvWaitKey(0);

	IplImage* origin_copy = cvCreateImage(cvSize(originSrc->width,originSrc->height),8,3);
	cvCopy(originSrc,origin_copy,NULL);
	Mat originCopyMat(origin_copy);
	for(int pt = 0;pt < detect_window_up.size();pt++)
	{		
		//cvRectangle(origin_copy,cvPoint(detect_window_up.at(pt).x-5,detect_window_up.at(pt).y-10),cvPoint(detect_window_down.at(pt).x+5,detect_window_down.at(pt).y+5),Scalar(0,255,255),1,8,0);
		cvSetImageROI(origin_copy, cvRect(detect_window_up.at(pt).x+5, detect_window_up.at(pt).y+5, 
			detect_window_down.at(pt).x-5-(detect_window_up.at(pt).x), detect_window_down.at(pt).y-5-(detect_window_up.at(pt).y)));
		IplImage *originROI = cvCreateImage(cvGetSize(origin_copy), origin_copy->depth,origin_copy->nChannels);
		cvCopy(origin_copy, originROI, NULL);
		Mat originROIMat(originROI);
		Mat originROIHSV;
		cvtColor(originROIMat,originROIHSV,COLOR_BGR2HSV); 
		//namedWindow("HSV ROI",1);
		//imshow("HSV ROI",originROIHSV);
		//waitKey(0);
		Mat imgClustered;
		Eigen::Vector3f colorVec;
			// ���ຯ��
			// ���룺hsvͼ��originROIHSV�������������ͼ��imgClustered��������׮Ͳ����ɫ������colorVec��
		colorVec = ColorCluster(originROIHSV, imgClustered); 
		//namedWindow("image after cluster",1);
		//imshow("image after cluster",imgClustered);
		//waitKey(0);
			// �ж���ɫ����
			// ���룺��ɫ����
			// ���أ���ɫ����
            int color_bucket = ExtractColor(colorVec);
            // color_bucket is the color of bucket which is under processing.
            // 0 -- blue
            // 1 -- yellow
            // 2 -- red
            // 3 -- unknown
            // 4 -- invalid
            
            Scalar color_rect;

            switch(color_bucket)
            {
                case 0:
                    color_rect = Scalar(255,0,0);
                    break;
                case 1:
                    color_rect = Scalar(0,255,255);
                    break;
                case 2:
                    color_rect = Scalar(0,0,255);
                    break;
                case 3:
                    color_rect = Scalar(100,100,100); 
                    break;
                case -1:
                    color_rect = Scalar(0,0,0); 
                    break;                                                   
            }
			// ��ͬ��ɫ��׮Ͳ����ͬ��ɫ�Ŀ�
            rectangle(originCopyMat,Point(detect_window_up.at(pt).x-5, detect_window_up.at(pt).y-10),Point(detect_window_down.at(pt).x+5, detect_window_down.at(pt).y+5),color_rect);
           
            char str_hue[10];
            sprintf(str_hue,"%d",(int)colorVec[0]);
			// ����Ӧ��Hͨ��ֵ��ʾ��ͼ���У��������
            putText(originCopyMat, str_hue, Point(detect_window_down.at(pt).x+5, detect_window_down.at(pt).y+5), FONT_HERSHEY_SIMPLEX, 1, color_rect, 1);
		    cvResetImageROI(origin_copy);
	}
	//cvNamedWindow("colorfinalresult",1);
	//cvShowImage("colorfinalresult",origin_copy);
	namedWindow("colorfinalresult",1);
	imshow("colorfinalresult",originCopyMat);
	waitKey(0);
  
	//cvReleaseImage( &src);  
	//cvReleaseImage( &sampleImg );  
	//cvReleaseImage( &tst );  
	//cvReleaseImage( &tst_tmp );  
	cvReleaseMat( &data_mat );  
	cvReleaseMat( &res_mat );  
  
	return 0;  
}  