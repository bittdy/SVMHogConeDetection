#define NEED_TRAIN 0

#include "cv.h"  
#include "highgui.h"  
#include <ml.h>  
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>  
#include <sstream>
#include <vector>
using namespace cv;  
using namespace std;  
  
int isOverLap(CvPoint P1,CvPoint P2,CvPoint PN1,CvPoint PN2)
{
	//return 0 when there is overlap otherwise return 1
	if(P1.x>PN2.x){return 1;}
	if(PN1.x>P2.x){return 1;}
	if(P2.y<PN1.y){return 1;}
	if(PN2.y<P1.y){return 1;}
	return 0;
}
  
int main(int argc, char** argv)    
{    
	int k=0;
	int j=0;
	int imageNumber = 10000;
	stringstream imageStream;
    vector<string> img_path;//输入文件名变量 
    vector<int> img_catg;  
	vector<CvPoint> detect_window_up;
	vector<CvPoint> detect_window_down;
    int nLine = 0;  
    string buf;  
    ifstream svm_data( "trainSample.txt" );//首先，这里搞一个文件列表，把训练样本图片的路径都写在这个txt文件中，使用bat批处理文件可以得到这个txt文件   
    unsigned long n;  
  
    while( svm_data )//将训练样本文件依次读取进来  
    {  
        if( getline( svm_data, buf ) )  
        {  
            nLine ++;  
            if( nLine % 2 == 0 )//这里的分类比较有意思，看得出来上面的SVM_DATA.txt文本中应该是一行是文件路径，接着下一行就是该图片的类别，可以设置为0或者1，当然多个也无所谓 
            {  
                 img_catg.push_back( atoi( buf.c_str() ) );//atoi将字符串转换成整型，标志（0,1），注意这里至少要有两个类别，否则会出错  
            }  
            else  
            {  
                img_path.push_back( buf );//图像路径  
            }  
        }  
    }  
    svm_data.close();//关闭文件  
  
    CvMat *data_mat, *res_mat;  
    int nImgNum = nLine / 2; //读入样本数量 ，因为是每隔一行才是图片路径，所以要除以2 
    ////样本矩阵，nImgNum：横坐标是样本数量， WIDTH * HEIGHT：样本特征向量，即图像大小  
    data_mat = cvCreateMat( nImgNum, 1764, CV_32FC1 );  //这里第二个参数，即矩阵的列是由下面的descriptors的大小决定的，可以由descriptors.size()得到，且对于不同大小的输入训练图片，这个值是不同的
    cvSetZero( data_mat );  
    //类型矩阵,存储每个样本的类型标志  
    res_mat = cvCreateMat( nImgNum, 1, CV_32FC1 );  
    cvSetZero( res_mat );  
  
    IplImage* src;  
    IplImage* trainImg=cvCreateImage(cvSize(64,64),8,1);//需要分析的图片，这里默认设定图片是64*64大小，所以上面定义了1764，如果要更改图片大小，可以先用debug查看一下descriptors是多少，然后设定好再运行  
  
	//开始搞HOG特征
    for( string::size_type i = 0; i != img_path.size(); i++ )  
    {  
            src=cvLoadImage(img_path[i].c_str(),0);  
            if( src == NULL )  
            {  
                cout<<" can not load the image: "<<img_path[i].c_str()<<endl;  
                continue;  
            }  
  
            cout<<" processing "<<img_path[i].c_str()<<endl;  
                 
            cvResize(src,trainImg);   //读取图片     
            HOGDescriptor *hog=new HOGDescriptor(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);  //具体意思见参考文章1,2     
            vector<float>descriptors;//结果数组     
            hog->compute(trainImg, descriptors,Size(1,1), Size(0,0)); //调用计算函数开始计算     
            cout<<"HOG dims: "<<descriptors.size()<<endl;  
            //CvMat* SVMtrainMat=cvCreateMat(descriptors.size(),1,CV_32FC1);  
            n=0;  
            for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)  
            {  
                cvmSet(data_mat,i,n,*iter);//把HOG存储下来  
                n++;  
            }  
                //cout<<SVMtrainMat->rows<<endl;  
            cvmSet( res_mat, i, 0, img_catg[i] );  
            cout<<" end processing "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;  
    }  
      
               
    CvSVM svm;//新建一个SVM    
    CvSVMParams param;//这里是参数
    CvTermCriteria criteria;    
    criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );    
    param = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );    
/*    
    SVM种类：CvSVM::C_SVC    
    Kernel的种类：CvSVM::RBF    
    degree：10.0（此次不使用）    
    gamma：8.0    
    coef0：1.0（此次不使用）    
    C：10.0    
    nu：0.5（此次不使用）    
    p：0.1（此次不使用）    
    然后对训练数据正规化处理，并放在CvMat型的数组里。    
                                                        */       
    //☆☆☆☆☆☆☆☆☆(5)SVM学习☆☆☆☆☆☆☆☆☆☆☆☆         
    svm.train( data_mat, res_mat, NULL, NULL, param );//训练啦    
    //☆☆利用训练数据和确定的学习参数,进行SVM学习☆☆☆☆     
    svm.save( "SVM_DATA.xml" );

	IplImage* originSrc = cvLoadImage("test.jpg",1);
	IplImage* test = cvCreateImage(cvSize(originSrc->width,originSrc->height),8,1);
	cvCvtColor(originSrc , test , CV_BGR2GRAY);//源图像和目标图像都必须是IPL_DEPTH_32F的
	
	IplImage* test_copy = cvCreateImage(cvSize(test->width,test->height),8,1);
	IplImage* test_copy1 = cvCreateImage(cvSize(test->width,test->height),8,1);
	IplImage* trainROI = cvCreateImage(cvSize(64,64),8,1);
	cvCopy(test,test_copy,NULL);
	cvCopy(test,test_copy1,NULL);
	for(k = 0; k < test->width-55; k=k+28)
	{
		for(j=0; j < test->height-110; j=j+55)
		{
			cvRectangle(test_copy1,cvPoint(k,j),cvPoint(k+55,j+110),Scalar(0,0,0),1,8,0);
			cvNamedWindow("imgNow",1);
			cvShowImage("imgNow",test_copy1);
			//cvWaitKey(0);
			cvSetImageROI(test, cvRect(k, j, 55, 110));
			IplImage *img = cvCreateImage(cvSize(64,64), test->depth,test->nChannels);
			cvResize(test,img);
			cvNamedWindow("imgDetecting",1);
			cvShowImage("imgDetecting",img);
			//cvWaitKey(0);
			cvResetImageROI(test);
			HOGDescriptor *hog1=new HOGDescriptor(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);  //具体意思见参考文章1,2     
			vector<float>descriptors1;//结果数组     
			hog1->compute(img, descriptors1,Size(1,1), Size(0,0)); //调用计算函数开始计算     
			cout<<"HOG dims: "<<descriptors1.size()<<endl;  
			CvMat* SVMtrainMat=cvCreateMat(1,descriptors1.size(),CV_32FC1);  
			n=0;  
			for(vector<float>::iterator iter1=descriptors1.begin();iter1!=descriptors1.end();iter1++)  
			{  
				cvmSet(SVMtrainMat,0,n,*iter1);  
				n++;  
			}    
			int ret = svm.predict(SVMtrainMat);//获取最终检测结果，这个predict的用法见 OpenCV的文档 
			if(ret == 1)
			{
				cvRectangle(test_copy,cvPoint(k,j),cvPoint(k+55,j+110),Scalar(0,255,255),1,8,0);//画出检测为桩桶的roi，方便确定训练负样本,并将之存下来
				imageNumber++;
				imageStream<<imageNumber;
				string imageName = "F:\\vs2010_projects\\OpencvSvm\\OpencvSvm\\hardexample\\"+imageStream.str()+".jpg";
				char *saveImageName = (char *)imageName.data();
				cvSaveImage(saveImageName,img);
				imageStream.str("");
				cvNamedWindow("img",1);
				cvShowImage("img",img);
 				//cvWaitKey(0);
				if(0==detect_window_up.size())
				{
					detect_window_up.push_back(cvPoint(k,j));
					detect_window_down.push_back(cvPoint(k+55,j+110));
					continue;
				}
				int vector_size = detect_window_up.size();
				for(int m=0;m<vector_size;m++)
				{
					if(0==isOverLap(cvPoint(k,j),cvPoint(k+55,j+110),detect_window_up.at(m),detect_window_down.at(m)))//说明和第m个框有重叠
					{
						CvPoint cvPointBuffer;
						cvPointBuffer.x = min(k,detect_window_up.at(m).x);
						cvPointBuffer.y = min(j,detect_window_up.at(m).y);
						detect_window_up.at(m).x = cvPointBuffer.x;
						detect_window_up.at(m).y = cvPointBuffer.y;
						cvPointBuffer.x = max(k+55,detect_window_down.at(m).x);
						cvPointBuffer.y = max(j+110,detect_window_down.at(m).y);
						detect_window_down.at(m).x = cvPointBuffer.x ;
						detect_window_down.at(m).y = cvPointBuffer.y;
						break;
					}
					if(m==vector_size-1)
					{
						detect_window_up.push_back(cvPoint(k,j));
						detect_window_down.push_back(cvPoint(k+55,j+110));
					}
				}
				
			}
			if(j+110>test->height-110)   //当当前测试点加上64大于图像宽时说明已经行检测完成，终止循环
				break;
		}
		if(k+55>test->width-55)   //当当前测试点加上64大于图像高时说明已经列检测完成，终止循环
			break;
	}
	cvNamedWindow("finalresult",1);
	cvShowImage("finalresult",test_copy);
	cvWaitKey(0);

	IplImage* origin_copy = cvCreateImage(cvSize(test->width,test->height),8,3);
	cvCopy(originSrc,origin_copy,NULL);
	for(int pt = 0;pt < detect_window_up.size();pt++)
	{
		cvRectangle(origin_copy,detect_window_up.at(pt),detect_window_down.at(pt),Scalar(0,255,255),1,8,0);
		//cvSetImageROI(origin_copy, cvRect(detect_window_up.at(pt).x, detect_window_up.at(pt).y, 
		//	detect_window_down.at(pt).x, detect_window_down.at(pt).y));
		//IplImage *originROI = cvCreateImage(cvGetSize(origin_copy), origin_copy->depth,origin_copy->nChannels);
		//cvCopy(origin_copy, originROI, NULL);
		//
		//cvResetImageROI(origin_copy);
	}
	cvNamedWindow("colorfinalresult",1);
	cvShowImage("colorfinalresult",origin_copy);
	cvWaitKey(0);
  
	//cvReleaseImage( &src);  
	//cvReleaseImage( &sampleImg );  
	//cvReleaseImage( &tst );  
	//cvReleaseImage( &tst_tmp );  
	cvReleaseMat( &data_mat );  
	cvReleaseMat( &res_mat );  
  
	return 0;  
}  