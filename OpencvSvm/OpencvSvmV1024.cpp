/*******************************
		图像课课设程序
		交通路锥识别
		刘欣，王小然，孙洋洋，田戴荧
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
	// kmeans聚类预处理
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

	// 获得聚类结果，并计算聚成的两类的面积
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
	
	// 聚成的两类的聚类重心
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
	// 判断条件
	// 1、小的部分的面积的二倍小于大的部分（可调整）
	// 2、小的部分的S通道大于35（可调整）
   if(-3!=case1 && -3!=case2)  //如果存在两个类均不为unknow，找出主类
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
		else //否则定位 invalid 属性，颜色向量全部置为0（有些欠妥，可以加一个属性，以便把颜色向量传回，方便调参）
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

//为了试车对程序改动最小，该函数仍保留，实际可以直接用上面的函数直接检测
int ExtractColor(const Eigen::Vector3f color) // 直接人工确定红黄蓝三个颜色区间，仅通过H通道判断。（参数可调整）
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
            if( nLine % 2 == 0 )//SVM_DATA.txt文本中应该是一行是文件路径，接着下一行就是该图片的类别，可以设置为0或者1，当然多个也无所谓 
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
    svm.train( data_mat, res_mat, NULL, NULL, param );//训练    
    //☆☆利用训练数据和确定的学习参数,进行SVM学习☆☆☆☆     
    svm.save( "SVM_DATA.xml" );

	IplImage* originSrc = cvLoadImage("test1.jpg",1);
	IplImage* test = cvCreateImage(cvSize(originSrc->width,originSrc->height),8,1);
	cvCvtColor(originSrc , test , CV_BGR2GRAY);//源图像和目标图像都必须是IPL_DEPTH_32F的
	
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
				cvRectangle(test_copy,cvPoint(k,j),cvPoint(k+63,j+63),Scalar(0,255,255),1,8,0);//画出检测为桩桶的roi，方便确定训练负样本,并将之存下来
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
					if((0==isOverLap(cvPoint(k,j),cvPoint(k+63,j+63),detect_window_up.at(m),detect_window_down.at(m)))&&(double(areaCount(P1_1,P1_2))/(64*64)>=0.4))//说明和第m个框有重叠且重叠面积达到一半面积
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
			if(j+63>test->height)   //当当前测试点加上64大于图像宽时说明已经行检测完成，终止循环
				break;
		}
		if(k+63>test->width)   //当当前测试点加上64大于图像高时说明已经列检测完成，终止循环
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
			// 聚类函数
			// 输入：hsv图像（originROIHSV），输出聚类结果图像（imgClustered），返回桩筒的颜色向量（colorVec）
		colorVec = ColorCluster(originROIHSV, imgClustered); 
		//namedWindow("image after cluster",1);
		//imshow("image after cluster",imgClustered);
		//waitKey(0);
			// 判断颜色函数
			// 输入：颜色向量
			// 返回：颜色种类
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
			// 不同颜色的桩筒画不同颜色的框
            rectangle(originCopyMat,Point(detect_window_up.at(pt).x-5, detect_window_up.at(pt).y-10),Point(detect_window_down.at(pt).x+5, detect_window_down.at(pt).y+5),color_rect);
           
            char str_hue[10];
            sprintf(str_hue,"%d",(int)colorVec[0]);
			// 将对应的H通道值显示在图像中，方便调参
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