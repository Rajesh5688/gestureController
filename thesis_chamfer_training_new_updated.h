#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

#define SIZE 10

typedef struct m{
    int y;
    int x;
    double chamf_D[20];
}Train_Models;

struct edge_coord{
    int x;
    int y;
    struct edge_coord * nxt;
};
typedef struct edge_coord EDGE;

EDGE *shape_template[20];


struct n1{
    int command;
    Train_Models models[SIZE];
};

typedef struct n1 train_data;

/***********MODIFICATIONS FOR SKIN DETECTION ****************/
static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;

CvRect selections;

const char *cascade_name=".\\data\\haarcascade_frontalface_alt2.xml";

/************************************************************/

float skin_hist[32][32][32],non_skin_hist[32][32][32];

void timer_setup()
{
    IplImage *img = cvCreateImage(cvSize(400,300),8,3);
    cvZero(img);
    /* initialize font and add text */
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, 1.0, 0, 2, 4);
    cvPutText(img, "Training Module !!!", cvPoint(50,(img->height/2)), &font, cvScalar(0, 0, 255, 0));
    cvNamedWindow("image", CV_WINDOW_AUTOSIZE);
    cvShowImage("image", img);
    cvWaitKey(2000);
    cvZero(img);
    cvInitFont(&font, CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, 1.0, 0, 1, 4);
    cvPutText(img, "Please be ready with your Gesture !!!", cvPoint(0,(img->height/2)), &font, cvScalar(0, 0, 255, 0));
    cvNamedWindow("image", CV_WINDOW_AUTOSIZE);
    cvShowImage("image", img);
    cvWaitKey(2000);
    cvZero(img);
    cvInitFont(&font, CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, 1.0, 0, 2, 4);
    cvPutText(img, "Ready!!!", cvPoint(50,(img->height/2)), &font, cvScalar(0, 0, 255, 0));
    cvNamedWindow("image", CV_WINDOW_AUTOSIZE);
    cvShowImage("image", img);
    cvWaitKey(2000);
    cvDestroyWindow("image");
    cvReleaseImage(&img);
}

IplImage * background_capture()
{
    IplImage *img = cvCreateImage(cvSize(400,300),8,3);
    cvZero(img);
    /* initialize font and add text */
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, 1.0, 0, 2, 4);
    cvPutText(img, "Capturing Background !!!", cvPoint(50,(img->height/2)), &font, cvScalar(0, 0, 255, 0));
    cvShowImage("BG",img);
    cvWaitKey(2000);
    cvDestroyWindow("BG");
    cvReleaseImage(&img);
    IplImage *img_capture;
    CvCapture *capture=cvCaptureFromCAM(0);
    img_capture=cvQueryFrame(capture);
    uchar *data= (uchar *) img_capture->imageData;
    while(data[0] == 0)
    {img_capture=cvQueryFrame(capture);}

    IplImage *img_bg=cvCreateImage(cvSize(img_capture->width,img_capture->height),8,1);
    IplImage *img_bg_grey=cvCreateImage(cvSize(img_capture->width/2,img_capture->height/2),8,1);
    //cvSaveImage("e:\\thesis_shapes\\outputs\\bg_capture.jpg",img_capture,0);
    cvCvtColor(img_capture,img_bg,CV_RGB2GRAY);
    cvResize(img_bg,img_bg_grey,1);
    cvReleaseImage(&img_bg);
    cvShowImage("rajeshhhh",img_bg_grey);
    cvReleaseCapture(&capture);
    return img_bg_grey;
}


void get_skin_hist()
{
    FILE *fp,*fp1;
    char buff[1024];
    char buff1[1024];
    fp=fopen(".\\data\\skin.txt","r");
    fp1=fopen(".\\data\\non_skin.txt","r");
    int k,k1,k2=0;

    while(!feof(fp) && !feof(fp1))
    {
        for (k=0;k<32;k++)
        {for(k1=0;k1<32;k1++)
            {
                fscanf(fp,"%s",buff);
                fscanf(fp1,"%s",buff1);
                skin_hist[k][k1][k2]=atof(buff);
                non_skin_hist[k][k1][k2]=atof(buff1);
                if (buff[0] > 57)
                {k2=k2+1;k1=0;};
            }
        }
    }
    fclose(fp);
    fclose(fp1);
}

IplImage* GetSkinScore(IplImage *img,float skin_hist[][32][32],float non_skin_hist[][32][32])
{
    //IplImage *imgHSV=cvCreateImage(cvGetSize(img),IPL_DEPTH_32F,3);
    IplImage * skin=cvCreateImage(cvGetSize(img),IPL_DEPTH_32F,1);
    //cvConvertScale(img,imgHSV,1.0/8.0,0);

    int height=skin->height;
    int width=skin->width;
    int step1=img->widthStep/sizeof(float);
    int step=skin->widthStep/sizeof(float);
    int channels=img->nChannels;
    float *data= (float *) skin->imageData;
    float *data1=(float *) img->imageData;
    float red,blue,green,tot;
    int r_index,g_index,b_index;
    for (int k=0;k<height;k++)
    {
        for (int k1=0;k1<width;k1++)
        {
            red=data1[k*step1+k1*channels+2];
            green=data1[k*step1+k1*channels+1];
            blue=data1[k*step1+k1*channels+0];

            r_index=(int) red/8;
            g_index=(int) green/8;
            b_index=(int) blue/8;
            tot=skin_hist[r_index][g_index][b_index]+non_skin_hist[r_index][g_index][b_index];
            if(tot != 0.0)
                data[k*step+k1]=skin_hist[r_index][g_index][b_index]/tot;
            else
                data[k*step+k1]=0.0;
        }
    }

    return skin;
}

double get_edge_sum(IplImage *test_template,EDGE *edge_cord)
{
    double dist=0.0;
    int step=test_template->widthStep/sizeof(float);
    float *data= (float *) test_template->imageData;
    EDGE *temp_edge = edge_cord->nxt;
    while (temp_edge->nxt != NULL)
    {
        dist=dist+data[(temp_edge->y)*step+temp_edge->x];
        temp_edge=temp_edge->nxt;
    }
    return dist;
}

Train_Models get_chamfer_score(IplImage *sub_window,EDGE **shape1)
{
    Train_Models t;
    float width_offset=40.0,height_offset=30.0;
    int k,k1,k2,flag=0;
    double dist;
    float width = sub_window->width,height = sub_window->height;
    CvRect rect;
    IplImage *test_template=cvCreateImage(cvSize(120,90),IPL_DEPTH_32F,1);
    float width_center=(float)((width-width_offset)/(test_template->width));
    width_center= ((test_template->width)/2)*width_center;
    float height_center=(float)((height - height_offset)/(test_template->height));
    height_center=((test_template->height)/2)*height_center;
    for (k=1;k<=width_offset;k++)
    {
        for (k1=1;k1<=height_offset;k1++)
        {
            cvSetImageROI(sub_window,cvRect(k,k1,width-width_offset,height-height_offset));
            cvResize(sub_window,test_template,1);
            for (k2=0;k2<2;k2++) // was 6 here
            {
                dist=get_edge_sum(test_template,shape1[k2]);
                if ((t.chamf_D[k2] > dist) && (flag == 1))
                {t.chamf_D[k2]=dist;
                    t.x=(int)width_center;
                    t.x=t.x+k;
                    t.y=(int)height_center;
                    t.y=t.y+k1;
                }
                if(flag==0)
                {
                    t.chamf_D[k2]=dist;
                    t.x=(int)width_center;
                    t.x=t.x+k;
                    t.y=(int)height_center;
                    t.y=t.y+k1;
                }
            }
            flag=1;
            cvResetImageROI(sub_window);
        }
    }
    return t;
}
Train_Models get_chamfer_dist(IplImage *dist_trans,EDGE **templ,CvPoint loc)
{
    int loop_flag=0,flag=0,x_ind,y_ind,k,k1;
    double chamfer_score,chamf_D=10000.0;
    float chamfer_scales[4]={1,0.75};
    CvRect coord;
    Train_Models result;

    if((loc.x)<=0)
    {coord.x=0;x_ind=160;}
    else if  ((loc.x > 0) && ((loc.x -80) <0))
    {coord.x=0;x_ind=160;}
    else
    {coord.x=loc.x-80;x_ind=160;}
    if((loc.y)<=0)
    {coord.y=0;y_ind =120;}
    else if  ((loc.y > 0) && ((loc.y -60) <0))
    {coord.y=0;y_ind=120;}
    else
    {coord.y=loc.y-60;y_ind=120;}
    if((coord.x+x_ind) > dist_trans->width)
        x_ind = dist_trans->width;
    if((coord.y+y_ind) > dist_trans->height)
        y_ind = dist_trans ->height;
    coord.height=y_ind;
    coord.width=x_ind;


    cvSetImageROI(dist_trans,coord);
    for (k=1;k<2;k++)
    {
        IplImage *sub_window=cvCreateImage(cvSize(160*chamfer_scales[k],120*chamfer_scales[k]),IPL_DEPTH_32F,1);
        cvResize(dist_trans,sub_window,1);
        result=get_chamfer_score(sub_window,templ);
        x_ind=result.x*(1/chamfer_scales[k]);
        y_ind=result.y*(1/chamfer_scales[k]);
    }
    cvResetImageROI(dist_trans);
    result.x=x_ind+coord.x;
    result.y=y_ind+coord.y;
    return result;
}

void writetofile(IplImage *img,IplImage *img1,IplImage *img2,IplImage *img3,IplImage *img4,IplImage *img5,int index)
{
    char raj[128]="e:\\thesis_shapes\\outputs\\new\\edges";
    char raj1[128]="e:\\thesis_shapes\\outputs\\new\\dist_transform";
    char raj2[128]="e:\\thesis_shapes\\outputs\\new\\framediff";
    char raj3[128]="e:\\thesis_shapes\\outputs\\new\\background";
    char raj4[128]="e:\\thesis_shapes\\outputs\\new\\inputframe";
    char raj5[128]="e:\\thesis_shapes\\outputs\\new\\output_new";
    int len=strlen(raj);
    raj[len]=(char)48+index-7;
    raj1[strlen(raj1)]=(char)48+index-7;
    raj2[strlen(raj2)]=(char)48+index-7;
    raj3[strlen(raj3)]=(char)48+index-7;
    raj4[strlen(raj4)]=(char)48+index-7;
    raj5[strlen(raj5)]=(char)48+index-7;
    strcat(raj,".jpg");
    strcat(raj1,".jpg");
    strcat(raj2,".jpg");
    strcat(raj3,".jpg");
    strcat(raj4,".jpg");
    strcat(raj5,".jpg");
    //cvSaveImage(raj,img,0);
    cvSaveImage(raj1,img1,0);
    //cvSaveImage(raj2,img2,0);
    //cvSaveImage(raj3,img3,0);
    //cvSaveImage(raj4,img4,0);
    cvSaveImage(raj5,img5,0);

}

train_data get_training_gesture_features(EDGE** shape)
{
    int gauss_flag=0,face_flag=0,frame_no=0, start_flag=0,k,k1;
    int feature_count=0;
    double min,max,max1,loc_chamf=0.0,crossloc_chamf=0.0;
    float interval;
    CvRect selections;
    CvPoint center;
    train_data t1;
    clock_t start;

    Train_Models* M=(Train_Models *) calloc(500,sizeof(Train_Models));
    int count_train_m=0;
    CvPoint pt1,pt2,max_point,min_point,tmp_pt;
    IplImage *imgframediff1=NULL;
    IplImage *imgframediff2=NULL;
    IplImage *imgframediff3 = NULL;
    IplImage *imgframediff=NULL;
    IplImage *skin_score = NULL;
    IplImage *frame_temp=NULL;
    IplImage *frame1=0;
    IplImage *frame3=0;
    IplImage* static_bg;
    static_bg = background_capture();
    timer_setup();

    CvCapture *capture=cvCaptureFromCAM(0);
    if(!capture)
    {printf("Could not initialize Capturing .. \n ");
        exit(1);
    }
    frame_temp = cvQueryFrame(capture);
    IplImage *frame_temp32f=cvCreateImage(cvGetSize(frame_temp),IPL_DEPTH_32F,3);
    IplImage *frame2=cvCreateImage(cvSize(frame_temp->width/2,frame_temp->height/2),IPL_DEPTH_32F,3);
    IplImage *frame_temp_resize=cvCreateImage(cvGetSize(frame2),8,3);
    IplImage *frame_temp_resize1=cvCreateImage(cvGetSize(frame2),8,3);
    cvCvtScale(frame_temp,frame_temp32f);
    cvResize(frame_temp32f,frame2);
    IplImage *bgsub=cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    IplImage *bg=cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    skin_score=cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    frame3 = cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,3);
    frame1 = cvCloneImage(frame2);
    imgframediff =  cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    imgframediff3 = cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    imgframediff2 = cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    imgframediff1 = cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    IplImage* edge1 = cvCreateImage(cvGetSize(frame2),8,1);
    IplImage* edge_32f = cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    IplImage *skin_result=cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    while (true)
    {

        frame_temp = cvQueryFrame(capture);
        cvResize(frame_temp,frame_temp_resize);
        cvCvtScale(frame_temp,frame_temp32f);
        cvResize(frame_temp32f,frame2);
        cvWaitKey(20);
        frame_temp = cvQueryFrame(capture);
        cvCvtScale(frame_temp,frame_temp32f);
        cvResize(frame_temp32f,frame3);

        if (face_flag ==0)
        {
            cvResize(frame_temp,frame_temp_resize,0);
            cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
            CvSeq* faces = cvHaarDetectObjects( frame_temp_resize, cascade, storage,1.1, 2, CV_HAAR_DO_CANNY_PRUNING,cvSize(40, 40) );
            CvRect* r = (CvRect *) cvGetSeqElem(faces,0);
            if(faces->total >0)
            {
                selections.x=r->x;
                selections.y=r->y;
                selections.width=r->width;
                selections.height=r->height;
                face_flag=1;
            }
            else
            {
                selections.x=0;
                selections.y=0;
                selections.height=2;
                selections.width=2;
            }
        }

        cvCvtColor(frame1,imgframediff1,CV_RGB2GRAY);
        cvCvtColor(frame2,imgframediff2,CV_RGB2GRAY);
        cvCvtColor(frame3,imgframediff3,CV_RGB2GRAY);

        cvAbsDiff(imgframediff1,imgframediff2,imgframediff);

        cvAbsDiff(imgframediff3,imgframediff2,imgframediff3);

        if(gauss_flag ==0)
        {//cvCopy(imgframediff2,bgsub);
            cvCvtScale(static_bg,bgsub,1,0);
            //cvCvtColor(frame_temp_resize,static_bg,CV_BGR2GRAY);
            gauss_flag=1;}

        cvAbsDiff(bgsub,imgframediff2,bg);
        cvConvertScale(bg,bg,1.0/255.0,0);

        //cvShowImage("Back Ground",bg);

        cvMin(imgframediff,imgframediff3,imgframediff);
        cvConvertScale(imgframediff,imgframediff,1.0/255.0,0);

        //cvShowImage("framediff",imgframediff);

        cvMinMaxLoc(imgframediff,&min,&max,&pt1,&pt2,0);

        printf(" The frame diff is %f \n", max);


        if ((max > 0.15) && (start_flag ==0))
        {
            start_flag=1;
        }

        if( start_flag == 1)
        {
            skin_score=GetSkinScore(frame2,skin_hist,non_skin_hist);
            cvSetImageROI(skin_score,selections);
            cvAndS(skin_score,cvScalarAll(0),skin_score,0);
            cvResetImageROI(skin_score);
            cvErode(skin_score,skin_score,0,1);
            cvDilate(skin_score,skin_score,0,1);
            cvShowImage("skin",skin_score);
            cvMul(skin_score,bg,skin_score,1);
            cvMul(skin_score,imgframediff,skin_result,1);
            cvShowImage("skin_bg",skin_score);
            cvCvtScale(skin_score,skin_score,255,0);
            cvMinMaxLoc(skin_result,&min,&max1,&min_point,&max_point,0);

            if (((max_point.x > selections.x) && (max_point.x < selections.width +selections.x)) && ((max_point.y > selections.y) && (max_point.y < selections.height +selections.y)))
                continue;

            if (max < 0.1)
            {
                frame_no=frame_no+1;
                if (frame_no ==1)
                    tmp_pt=max_point;
            }
            else
                frame_no=0;

            cvCvtColor(frame_temp_resize,edge1,CV_BGR2GRAY);
            cvAbsDiff(edge1,static_bg,edge1);
            cvCanny(edge1,edge1,50.0,100.0,3);
            cvShowImage("Edge",edge1);
            cvNot(edge1,edge1);
            cvDistTransform(edge1,edge_32f,2,3,0,0);
            cvMinMaxLoc(edge_32f,&min,&max1,&pt1,&pt2,0);
            //cvConvertScale(edge_32f,edge_32f,1.0/max1,0);
            cvShowImage("Dist Transform",edge_32f);
            start = clock();
            M[count_train_m]=get_chamfer_dist(edge_32f,shape,max_point);
            //printf(" CASE 'M'******** The count_train_m is %d value of x is %d and y is %d \n",count_train_m,M[count_train_m].x,M[count_train_m].y);
            printf(" The time of Execution was %f \n",((double) clock() - start) / CLOCKS_PER_SEC);
            feature_count=feature_count+1;
            //printf(" CASE MAX_POINT******** The count_train_m is %d value of x is %d and y is %d \n",count_train_m,max_point.x,max_point.y);
            pt1.x=M[count_train_m].x-20; pt1.y=M[count_train_m].y-20;
            if (pt1.x<0)
                pt1.x=0;
            if (pt1.y<0)
                pt1.y=0;
            pt2.x=M[count_train_m].x+20; pt2.y=M[count_train_m].y+20;
            if (pt2.x>frame2->width)
                pt2.x=frame2->width;
            if (pt2.y>frame2->height)
                pt2.y=frame2->height;
            cvRectangle( frame_temp_resize, pt1, pt2, CV_RGB(230,20,232), 3, 8, 0 );
            //cvRectangle(frame_temp_resize,cvPoint(max_point.x-20,max_point.y-20),cvPoint(max_point.x+20,max_point.y+20),CV_RGB(0,0,255),2,8,1);
            //cvRectangle( frame_temp_resize, cvPoint(max_point.x-60,max_point.y-45), cvPoint(max_point.x+60,max_point.y+45), CV_RGB(230,20,232), 3, 8, 0 );
            //cvRectangle( frame_temp_resize, cvPoint(max_point.x-40,max_point.y-30), cvPoint(max_point.x+40,max_point.y+30), CV_RGB(230,20,232), 3, 8, 0 );
            count_train_m=count_train_m+1;
            //if ((count_train_m >=5) && (count_train_m <= 10))
            //writetofile(edge1,edge_32f,imgframediff,bg,frame_temp,frame_temp_resize,count_train_m);
        }
        cvShowImage("name",frame_temp_resize);

        if( frame_no == 10)
        {
            //frame_no = (count_train_m)/20;
            feature_count=count_train_m-(frame_no+2);
            frame_no=4;

            for(k=0;k<SIZE;k++)
            {
                t1.models[k].x=M[k+frame_no].x;
                t1.models[k].y=M[k+frame_no].y;
                for(k1=0;k1<2;k1++)
                {t1.models[k].chamf_D[k1]=M[k+frame_no].chamf_D[k1];}
            }

            free(M);

            cvDestroyWindow("name");
            cvDestroyWindow("framediff");
            cvDestroyWindow("Back Ground");
            cvDestroyWindow("green_color");
            cvDestroyWindow("Dist Transform");
            cvDestroyWindow("Edge");
            cvReleaseImage(&frame1);
            cvReleaseImage(&frame2);
            cvReleaseImage(&frame3);
            cvReleaseImage(&imgframediff);
            cvReleaseImage(&imgframediff1);
            cvReleaseImage(&imgframediff2);
            cvReleaseImage(&skin_score);
            cvReleaseImage(&skin_result);
            cvReleaseImage(&frame_temp32f);
            cvReleaseImage(&frame_temp_resize);
            cvReleaseImage(&frame_temp_resize1);
            cvReleaseImage(&bg);
            cvReleaseImage(&bgsub);
            cvReleaseImage(&edge1);
            cvReleaseImage(&edge_32f);
            cvReleaseImage(&static_bg);
            cvReleaseCapture(&capture);
            break;
        }

        cvCopy(frame3,frame1);
    }
    return t1;
}

void get_edge_locations(EDGE **root,IplImage *shape)
{
    CvPoint minedge_coord,maxedge_coord;
    CvScalar s;
    s.val[0]=0;
    double minedge,maxedge;
    EDGE *temp_root = (*root);
    cvMinMaxLoc(shape,&minedge,&maxedge,&minedge_coord,&maxedge_coord,0);
    EDGE *e_t= (EDGE *) (malloc(sizeof(EDGE)));
    e_t->x=maxedge_coord.x;
    e_t->y=maxedge_coord.y;
    e_t->nxt=NULL;
    temp_root=e_t;
    while ((maxedge != 0.0) && (maxedge > 100))
    {
        EDGE *e_t= (EDGE *) (malloc(sizeof(EDGE)));
        e_t->x=maxedge_coord.x;
        e_t->y=maxedge_coord.y;
        e_t->nxt=temp_root;
        temp_root=e_t;
        cvSet2D(shape,maxedge_coord.y,maxedge_coord.x,s);
        cvMinMaxLoc(shape,&minedge,&maxedge,&minedge_coord,&maxedge_coord,0);
    }
    (*root)=temp_root;
}

void plotandcheck(EDGE *ed)
{
    IplImage *check=cvCreateImage(cvSize(120,90),8,1);
    cvZero(check);
    EDGE *temp_edge = ed;
    int step=check->widthStep/sizeof(uchar);
    uchar *data= (uchar *) check->imageData;
    while (temp_edge->nxt != NULL)
    {
        data[(temp_edge->y)*step+temp_edge->x]=255;
        temp_edge=temp_edge->nxt;
    }
    cvShowImage("check",check);
}

void get_hand_shape(EDGE **edge_coord,int index)
{
    IplImage *template_img = cvCreateImage(cvSize(640,480),8,3);
    CvCapture *capture=NULL;
    IplImage *hand_shape_template=cvCreateImage(cvSize(240,180), 8, 1);
    IplImage *hand_shape=cvCreateImage(cvSize(240,180), 8, 1);
    IplImage *h_s=cvCreateImage(cvSize(120,90), 8, 1);
    capture=cvCaptureFromCAM(0);
    //cvWaitKey(700);
    //IplImage *img_bg=cvCreateImage(cvSize(template_img->width,template_img->height),8,3);
    //img_bg = cvQueryFrame(capture);
    while(true)
    {
        template_img = cvQueryFrame(capture);
        if (cvWaitKey(100) != -1)
            break;
        cvRectangle( template_img, cvPoint(120,90), cvPoint(360,270) , CV_RGB(230,20,232), 3, 8, 0 );
        cvShowImage("Object",template_img);
    }
    //cvAbsDiff(template_img,img_bg,template_img);
    cvSetImageROI(template_img,cvRect(120,90,240,180));
    //cvShowImage("cropped_img",template_img);
    cvCvtColor(template_img,hand_shape_template,CV_BGR2GRAY);
    cvCanny(hand_shape_template,hand_shape,50.0,100.0,3);
    cvResize(hand_shape,h_s,1);
    cvShowImage("canny Image ",h_s);
    //cvSaveImage(raj,h_s,0);
    //cvSaveImage(raj1,template_img,0);
    //cvConvertScale(h_s,h_s,1/255,0);
    //cvDestroyWindow("cropped_img");
    get_edge_locations(edge_coord,h_s);
    cvDestroyWindow("Object");
    cvResetImageROI(template_img);
    //cvReleaseImage(&img_bg);
    cvReleaseCapture(&capture);
    cvReleaseImage(&hand_shape_template);
    cvReleaseImage(&h_s);
    cvReleaseImage(&hand_shape);
}

void training_error_msg(int k)
{
    CvFont font;
    IplImage *screen = cvCreateImage(cvSize(400,300),8,3);
    cvZero(screen);
    cvInitFont(&font, CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, 1.0, 0, 2, 4);
    if (k == 1)
        cvPutText(screen, "Training Gestures are Confusing, use another gesture !!!", cvPoint(0,(screen->height/2)), &font, cvScalar(0, 0, 255, 0));
    if (k == 2)
        cvPutText(screen, "Shapes scores are similar, another shape !!!", cvPoint(0,(screen->height/2)), &font, cvScalar(0, 0, 255, 0));
    cvNamedWindow("Training Redo", CV_WINDOW_AUTOSIZE);
    cvShowImage("Training Redo", screen);
    cvWaitKey(5000);
    cvReleaseImage(&screen);
    cvDestroyWindow("Training Redo");
}

train_data chamfer_training_module(int k,EDGE **shape)
{
    train_data training_data;
    training_data=get_training_gesture_features(shape);
    return training_data;
}
