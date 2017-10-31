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
    vector<string> img_path;//输入文件名变量 
    vector<int> img_catg;  
    int nLine = 0;  
    string buf;  
    ifstream svm_data( "data.txt" );//首先，这里搞一个文件列表，把训练样本图片的路径都写在这个txt文件中，使用bat批处理文件可以得到这个txt文件   
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
    data_mat = cvCreateMat( nImgNum, 365904, CV_32FC1 );  //这里第二个参数，即矩阵的列是由下面的descriptors的大小决定的，可以由descriptors.size()得到，且对于不同大小的输入训练图片，这个值是不同的
    cvSetZero( data_mat );  
    //类型矩阵,存储每个样本的类型标志  
    res_mat = cvCreateMat( nImgNum, 1, CV_32FC1 );  
    cvSetZero( res_mat );  
  
    IplImage* src; 
string path;
    IplImage* trainImg=cvCreateImage(cvSize(36,136),8,3);//需要分析的图片，这里默认设定图片是64*64大小，所以上面定义了1764，如果要更改图片大小，可以先用debug查看一下descriptors是多少，然后设定好再运行  
  
 vector<float>descriptors;//结果数组  
//开始搞HOG特征
    HOGDescriptor *hog=new HOGDescriptor(cvSize(16,16),cvSize(8,8),cvSize(8,8),cvSize(4,4),9);  //具体意思见参考文章1,2     

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
                 
            cvResize(src,trainImg);   //读取图片     
           
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
//{
//	vector<float>().swap(descriptors); //释放vector内存
//}
            cout<<" end processing "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl; 
cvReleaseImage( &src );
src=NULL;

    } 
cout<<endl;
    cout<<"【1】Hog提取特征结束，开始训练样本！"<<endl; 
cout<<endl;
               
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
           
    svm.train( data_mat, res_mat, NULL, NULL, param );//训练啦    
    svm.save( "SVM.xml" );
  
cout<<endl;
cout<<"【2】训练结束，以保存训练数据：SVM.xml"<<endl; 
cout<<endl;

    //检测样本  
    IplImage *test;  
    vector<string> img_tst_path;  
    ifstream img_tst( "test.txt" );//同输入训练样本，这里也是一样的，只不过不需要标注图片属于哪一类了
    while( img_tst )  
    {  
        if( getline( img_tst, buf ) )  
        {  
            img_tst_path.push_back( buf );  
        }  
    }  
    img_tst.close();  
  
  
cout<<endl;
cout<<"【3】开始测试样本！"<<endl; 
cout<<endl;

    CvMat *test_hog = cvCreateMat( 1, 365904, CV_32FC1 );//注意这里的1764，同上面一样 
CvMat* SVMtrainMat;
    char line[512];  
    ofstream predict_txt( "preditcion.txt" );//把预测结果存储在这个文本中  
CvSVM svm_hog;
svm_hog.load("SVM.xml");

    for( string::size_type j = 0; j != img_tst_path.size(); j++ )//依次遍历所有的待检测图片  
    {  
        path=img_tst_path[j];
test = cvLoadImage(path.c_str(), 1);  
        if( test == NULL )  
        {  
             cout<<" can not load the image: "<<img_tst_path[j].c_str()<<endl;  
               continue;  
         }  
          
        cvZero(trainImg);  
        cvResize(test,trainImg);   //读取图片     
       //vector<float>descriptors;//结果数组     
        hog->compute(trainImg, descriptors,Size(1,1), Size(0,0)); //调用计算函数开始计算     
       
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
  
        int ret = svm_hog.predict(SVMtrainMat);//获取最终检测结果，这个predict的用法见 OpenCV的文档 
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