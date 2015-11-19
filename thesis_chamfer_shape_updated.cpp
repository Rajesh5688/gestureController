#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <windows.h>
#include<iostream>

#include "SendKeys1.h"
#include "thesis_chamfer_training_new_updated.h"

unsigned int command_count=0;
unsigned int models_count=48;
static int detections=5;
unsigned int shape_flag=0;
char commands[40][100]={"up","down","left","right"};
char commands1[10][100]={"firstTrain","secondTrain","super","Victory","four_fing","camera","out","shape_left"};

struct n{
    double cum_D;
    int model_frame;
    long start_frame;
    long query_frame;
    int start_flag;
};

typedef struct n List;


double thresh=500.0;
const double train_thresh = 500.0;
double chamfer_dist=0.0,cross_chamfer_dist=0.0;

Train_Models models[SIZE+SIZE+28+8][SIZE];

Train_Models feature;

List ind_prev[SIZE+SIZE+28+8][SIZE] , ind_cur[SIZE+SIZE+28+8][SIZE];

CvPoint min_point[5],max_point[5];

HWND hnd;
CSendKeys m_sk;

void initialize_list()
{
    for (int k=0;k<models_count;k++)
    {
        //ind_prev[k]=(List **) calloc(SIZE,sizeof(List *));
        //ind_cur [k]= (List **) calloc(SIZE,sizeof(List *));
        ind_prev[k][0].start_flag=0;
        //models [k]= (Train_Models *) calloc(SIZE,sizeof(Train_Models));
    }
}

void get_training_trajectory()
{
    FILE *fp;
    int index=1;
    int k=0,count=-1,k1=0;
    char buff[1024];

    while(index <= 4)
    {
        char in_file[1024]=".\\data\\thesis_shapes\\traject";
        in_file[strlen(in_file)]=(char)48+index;
        strcat(in_file,".txt");
        fp=fopen(in_file,"r");
        if(!fp)
        {printf("Could Not Open File !!!! \n"); exit(1);}
        index=index+1;
        while(!feof(fp))
        {
            fscanf(fp,"%s",buff);
            if (buff[0] > 57)
            {count=count+1;k=0;k1=0;}
            else
            { if((k1 % 2) == 0)
                    models[count][k].y=atoi(buff);
                else
                {
                    models[count][k].x=atoi(buff);
                    k=k+1;
                }
                k1=k1+1;
            }
        }
        fclose(fp);
    }
    count=count+1;
    while(count<models_count)
    {
        for (k=0;k<SIZE;k++)
        {
            models[count][k].x=models[(count % 8)][k].x;
            models[count][k].y=models[(count % 8)][k].y;
        }
        count=count+1;
    }
}

void get_shape(int index)
{
    char location[128]=".\\data\\thesis_shapes\\shape_edge";
    location[strlen(location)]=(char) 49+index;
    strcat(location,".jpg");
    IplImage *img=cvLoadImage(location,0);
    get_edge_locations(&shape_template[index],img);
    if (index==0)
        plotandcheck(shape_template[index]);
}
void get_training_shapes()
{
    int k;
    for(k=0;k<2;k++)
    {get_shape(k);}
}


void get_models(train_data t1)
{
    int k,k1;
    for(k=0;k<SIZE;k++)
    {
        models[models_count][k].x=t1.models[k].y;
        models[models_count][k].y=t1.models[k].x;
        for (k1=0;k1<SIZE;k1++)
        {models[models_count][k].chamf_D[k1]=t1.models[k].chamf_D[k1];}
    }
    models_count=models_count+1;
}


void gesture_spotting(int frame_no,Train_Models shape_dist)
{	
    int k,k1,m,flag,start_frame,best_k=0;
    double score[5],temp,temp_thresh,chamf_score;
    char a[512];
    for (k=1;k<=models_count;k++) // was models_count here in condition
    {
        chamf_score=shape_dist.chamf_D[(k-1)/8];
        if (ind_prev[k-1][0].start_flag == 0)
        {
            ind_prev[k-1][0].start_flag=1;
            for(k1=1;k1<=SIZE;k1++)
            {
                /*List *new_list = (List *) malloc(sizeof(List));*/
                ind_prev[k-1][k1-1].model_frame=k1;
                ind_prev[k-1][k1-1].query_frame=frame_no;
                ind_prev[k-1][k1-1].start_frame=frame_no;
                ind_prev[k-1][k1-1].cum_D=chamf_score;
                //for (m=0;m<detections;m++)
                //{
                temp= sqrt((pow((double)(models[k-1][k1-1].x - shape_dist.y),2)) + (pow((double)(models[k-1][k1-1].y - shape_dist.x),2)));
                //if ((m==0) || (temp>score[m]))
                //temp=score[m];
                //}
                if ( (k1-1) ==0)
                    ind_prev[k-1][k1-1].cum_D=temp+ind_prev[k-1][k1-1].cum_D;
                else
                    ind_prev[k-1][k1-1].cum_D=ind_prev[k-1][k1-2].cum_D+temp+ind_prev[k-1][k1-1].cum_D;
            }
        }
        else
        {
            for (k1=1;k1<=SIZE;k1++)
            {
                // List *new_list = (List *) malloc(sizeof(List));
                ind_cur[k-1][k1-1].model_frame=k1;
                ind_cur[k-1][k1-1].query_frame=frame_no;
                ind_cur[k-1][k1-1].cum_D=chamf_score;
                //for (m=0;m<detections;m++)
                //{
                //score[m]= sqrt((pow((double)(models[k-1][k1-1].x - max_point[m].y),2)) + (pow((double)(models[k-1][k1-1].y - max_point[m].x),2)));
                temp= sqrt((pow((double)(models[k-1][k1-1].x - shape_dist.y),2)) + (pow((double)(models[k-1][k1-1].y - shape_dist.x),2)));
                //if ((m==0) || (temp>score[m]))
                //temp=score[m];
                //}
                if ( (k1-1)==0)
                {
                    ind_cur[k-1][k1-1].cum_D=temp+ind_cur[k-1][k1-1].cum_D;
                    ind_cur[k-1][k1-1].start_frame=frame_no;
                }
                else
                {
                    (ind_cur[k-1][k1-2].cum_D < ind_prev[k-1][k1-2].cum_D)?((ind_cur[k-1][k1-2].cum_D < ind_prev[k-1][k1-1].cum_D)?flag=1:flag=3):(ind_prev[k-1][k1-2].cum_D < ind_prev[k-1][k1-1].cum_D?flag=2:flag=3);
                    if(flag==1)
                    {
                        ind_cur[k-1][k1-1].cum_D=ind_cur[k-1][k1-2].cum_D+temp+ind_cur[k-1][k1-1].cum_D;
                        ind_cur[k-1][k1-1].start_frame=ind_cur[k-1][k1-2].start_frame;
                        //new_list->nxt=ind_cur[k-1][k1-2];
                        //ind_cur[k-1][k1-1]=new_list;
                    }
                    else if(flag==2)
                    {
                        ind_cur[k-1][k1-1].cum_D=ind_prev[k-1][k1-2].cum_D+temp+ind_cur[k-1][k1-1].cum_D;
                        ind_cur[k-1][k1-1].start_frame=ind_prev[k-1][k1-2].start_frame;
                        //new_list->nxt=ind_prev[k-1][k1-2];
                        //ind_cur[k-1][k1-1]=new_list;
                    }
                    else
                    {
                        ind_cur[k-1][k1-1].cum_D=ind_prev[k-1][k1-1].cum_D+temp+ind_cur[k-1][k1-1].cum_D;
                        ind_cur[k-1][k1-1].start_frame=ind_prev[k-1][k1-1].start_frame;
                        //new_list->nxt=ind_prev[k-1][k1-1];
                        //ind_cur[k-1][k1-1]=new_list;
                    }
                    ind_prev[k-1][k1-2].cum_D=ind_cur[k-1][k1-2].cum_D;
                    ind_prev[k-1][k1-2].start_frame=ind_cur[k-1][k1-2].start_frame;
                }
            }
            ind_prev[k-1][k1-2].cum_D=ind_cur[k-1][k1-2].cum_D;
            ind_prev[k-1][k1-2].start_frame=ind_cur[k-1][k1-2].start_frame;
            if (ind_cur[k-1][k1-2].cum_D <= thresh)
            {
                if (best_k==0)
                {
                    best_k=k;
                    temp_thresh=ind_cur[k-1][k1-2].cum_D;
                }
                if ((best_k!=0) && (temp_thresh > ind_cur[k-1][k1-2].cum_D))
                {
                    temp_thresh=ind_cur[k-1][k1-2].cum_D;
                    best_k=k;
                }
                //start_frame=back_track(ind_cur[best_k-1][k1-2]);
                //printf (" The Model is %d ..  frame diff is %d\n",k,frame_no-start_frame);
            }
        }
    }
//    printf (" The best value is %d\n", best_k-1);
    if ((best_k !=0) && ((ind_cur[best_k-1][k1-2].query_frame - ind_cur[best_k-1][k1-2].start_frame) >= 5))
    {
        printf (" Yes it is greater The model is %s .. %s frame diff is %d\n",commands1[(best_k-1)/8],commands[(((best_k-1)%8)/2)], ind_cur[best_k-1][k1-2].query_frame-ind_cur[best_k-1][k1-2].start_frame);
        hnd=GetForegroundWindow();
        GetWindowText(hnd,a,1024);
        if (!m_sk.AppActivate((LPCTSTR)a))
        {
            printf("Could not send to this application!\n");
            exit(1);
        }
        //DOWN
        char result[100];
        strcat(result, commands1[((best_k-1)%8)]);
        strcat(result, commands[(((best_k-1)%8)/2)]);
        strcat(result, "\n");
        m_sk.SendKeys((LPCTSTR)result, 0);  //for training_set1.txt
        //	m_sk.SendKeys((LPCTSTR)commands[(best_k-1)],0);
        best_k=0;
        for (int k=0;k < models_count;k++)
        {
            ind_prev[k][0].start_flag=0;
        }
    }
}

/*int check_shape_similarity(train_data *t)
{
    int k,k1,count=0;
    for(k=0;k<SIZE;k++)
    {
        for (k1=0;k1<SIZE;k++)
        {
        if((t->models[k].chamf_D - t->models[k].chamf_D) < 4)
            count=count+1;
        if(count >=4)
            return 1;
        }
    }
    return 0;
}*/

int check_similarity(int index)
{
    int k,k1,model_no,k2;
    double cum_D=0.0,diag,right,down;
    for (model_no=0;model_no<index;model_no++)
    {
        for (k2=0;k2<SIZE;k2++)
        {
            if(k2 !=index)
            {
                k=0;k1=0;
                cum_D=sqrt((pow((double)(models[model_no][k].y - models[index][k1].y),2)) + (pow((double)(models[model_no][k].x - models[index][k1].x),2)))+(models[index][k1].chamf_D[k2]);
                while((k<SIZE) && (k1<SIZE))
                {
                    diag=sqrt((pow((double)(models[model_no][k+1].y - models[index][k1+1].y),2)) + (pow((double)(models[model_no][k+1].x - models[index][k1+1].x),2)))+(models[index][k1+1].chamf_D[k2]);
                    right=sqrt((pow((double)(models[model_no][k].y - models[index][k1+1].y),2)) + (pow((double)(models[model_no][k].x - models[index][k1+1].x),2)))+(models[index][k1+1].chamf_D[k2]);
                    down=sqrt((pow((double)(models[model_no][k+1].y - models[index][k1].y),2)) + (pow((double)(models[model_no][k+1].x - models[index][k1].x),2)))+(models[index][k1].chamf_D[k2]);
                    if((diag<=right) && (diag <= down))
                    {
                        cum_D=cum_D+diag;
                        k=k+1;k1=k1+1;
                    }
                    else if ((right<diag) && (right < down))
                    {
                        cum_D=cum_D+right;
                        k=k;k1=k1+1;
                    }
                    else
                    {
                        cum_D=cum_D+down;
                        k=k+1;k1=k1;
                    }
                    if ((k == SIZE-1) && ((k1+1) != SIZE-1))
                    {
                        k1=k1+1;
                        cum_D=cum_D+sqrt((pow((double)(models[model_no][k].y - models[index][k1].y),2)) + (pow((double)(models[model_no][k].x - models[index][k1].x),2)))+(models[index][k1].chamf_D[k2]);
                    }
                    if (((k+1) != SIZE-1) && (k1 == SIZE-1))
                    {
                        k=k+1;
                        cum_D=cum_D+sqrt((pow((double)(models[model_no][k].y - models[index][k1].y),2)) + (pow((double)(models[model_no][k].x - models[index][k1].x),2)))+(models[index][k1].chamf_D[k2]);
                    }
                }
            }
            if (cum_D < train_thresh)
                return 1;
        }
        //if (cum_D < thresh*2)
        //	thresh=cum_D/2;

    }
    return 0;
}

void remove_shape_points(EDGE *ed)
{
    EDGE *temp=ed;
    while (ed->nxt!= NULL)
    {
        ed=ed->nxt;
        free(temp);
        temp=ed;
    }

}

void training_module()
{
    int k,result;
    train_data training_datas[10];
    for(k=0;k<2;k++)
    {get_hand_shape(&shape_template[k],k+1);}
    //plotandcheck(shape_template[k]);
    for (k=0;k<2;k++)
    {
        training_datas[k]=chamfer_training_module(k,shape_template);
        /*if(check_shape_similarity(&training_datas[k],k))
    {
    training_error_msg(2);
    remove_shape_points(shape_template[k]);
    get_hand_shape(&shape_template[k]);
    k=k-1;
    continue;
    }*/
        get_models(training_datas[k]);
        /*if (k != 0)
    {result=check_similarity(k);
    if (result)
    {
    training_error_msg(1);
    k=k-1;
    models_count=models_count-1;
    }
    }*/
    }
}


int main()
{
    int k,k1;
    static int c=-1;
    int frame_no=0,gauss_flag=0,face_flag=0;
    double min,max;
    clock_t start;
    storage = cvCreateMemStorage(0);
    CvPoint center;  //Modificatio
    CvPoint pt1,pt2;
    IplImage *imgTest = NULL;
    initialize_list();
    get_skin_hist();

    training_module();
//    get_training_trajectory();
//    get_training_shapes();
    std::cout << " FINISHED TRAINING ....." << std::endl;
    IplImage* static_bg = background_capture();

    //timer_setup();
    //exit(0);

    IplImage *imgframediff1=NULL;
    IplImage *imgframediff2=NULL;
    IplImage *imgframediff3 = NULL;
    IplImage *imgframediff=NULL;
    IplImage *skin_score = NULL;
    IplImage *sample_face=NULL;
    IplImage *frame_temp=NULL;
    IplImage *frame1=0;
    IplImage *frame3=0;
    IplImage *face_frame=0;

    CvCapture *capture=NULL;
    capture=cvCaptureFromCAM(0);
    if(!capture)
    {printf("Could not initialize Capturing .. \n ");
        return -1;
    }
    frame_temp = cvQueryFrame(capture);
    std::cout << " Frame width : " << frame_temp->width << " , height : " << frame_temp->height << std::endl;
    //            while(true) {
    //                imgTest = cvQueryFrame(capture);
    //                cvShowImage("Debug", imgTest);
    //                int c = cvWaitKey(10);
    //                if( c == 27) {
    //                    break;
    //                }
    //            }
    //            return 0;
    IplImage *frame_temp32f=cvCreateImage(cvGetSize(frame_temp),IPL_DEPTH_32F,3);
    IplImage *frame2=cvCreateImage(cvSize(frame_temp->width/2,frame_temp->height/2),IPL_DEPTH_32F,3);
    IplImage *frame_temp_resize=cvCreateImage(cvGetSize(frame2),8,3);
    cvCvtScale(frame_temp,frame_temp32f);
    cvResize(frame_temp32f,frame2);
    //CvMat *gauss=cvCreateMat(frame2->height,frame2->width,CV_32F);
    IplImage *bgsub=cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    IplImage *bg=cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    skin_score=cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    frame3 = cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,3);
    frame1 = cvCloneImage(frame2);
    imgframediff =  cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    imgframediff3 = cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    imgframediff2 = cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    imgframediff1 = cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    imgframediff3 = cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    IplImage* edge1 = cvCreateImage(cvGetSize(frame2),8,1);
    IplImage* edge_32f = cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    IplImage *skin_result=cvCreateImage(cvGetSize(frame2),IPL_DEPTH_32F,1);
    while (true)
    {

        frame_temp = cvQueryFrame(capture);
        cvResize(frame_temp,frame_temp_resize);
        cvCvtScale(frame_temp,frame_temp32f);
        cvResize(frame_temp32f,frame2);
        c=cvWaitKey(20);
        frame_temp = cvQueryFrame(capture);
        cvCvtScale(frame_temp,frame_temp32f);
        cvResize(frame_temp32f,frame3);

        if (frame_no == 65000)
            frame_no=0;
        //cvShowImage("name",frame_temp_resize);

        //if ((frame_no % 2) == 0)
        //{
        cvCvtColor(frame1,imgframediff1,CV_RGB2GRAY);
        cvCvtColor(frame2,imgframediff2,CV_RGB2GRAY);
        cvCvtColor(frame3,imgframediff3,CV_RGB2GRAY);

        cvAbsDiff(imgframediff1,imgframediff2,imgframediff);

        //cvConvertScale(imgframediff,imgframediff,2,0);

        cvAbsDiff(imgframediff3,imgframediff2,imgframediff3);

        if(gauss_flag==0)
        {
            //cvCopy(imgframediff2,bgsub);
            cvCvtScale(static_bg,bgsub,1,0);
            //cvCvtColor(frame_temp_resize,static_bg,CV_BGR2GRAY);
            //cvCanny(static_bg,static_bg,50.0,52.0,3);
            gauss_flag=1;
        }


        cvAbsDiff(bgsub,imgframediff2,bg);
        cvConvertScale(bg,bg,1.0/255.0,0);

        cvMin(imgframediff,imgframediff3,imgframediff);
        cvConvertScale(imgframediff,imgframediff,1.0/255.0,0);

        cvCvtColor(frame_temp_resize,edge1,CV_BGR2GRAY);
        //			cvCvtColor(frame_temp_resize,edge2,CV_BGR2GRAY);

        //cvCanny(edge1,edge1,50.0,50.0,3);
        //cvSub(edge1,static_bg,edge1);

        //value_check(imgframediff);

        //cvShowImage("base", static_bg);


        //cvShowImage("framediff",imgframediff);

        if (face_flag ==0)
        {
            cvResize(frame_temp,frame_temp_resize,0);
            cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
            CvSeq* faces = cvHaarDetectObjects( frame_temp_resize, cascade, storage,1.1, 2, CV_HAAR_DO_CANNY_PRUNING,cvSize(40, 40) );
            CvRect* r = (CvRect *) cvGetSeqElem(faces,0);
            if(faces->total >0)
            {
                center.x=cvRound(r->x+(r->width*0.5));
                center.y=cvRound(r->y+(r->height*0.5));
                selections.x=center.x-75;
                selections.y=center.y-75;
                if (selections.x < 0)
                    selections.x=0;
                if (selections.y < 0)
                    selections.y=0;
                selections.width=150;
                if (selections.width > frame2->width)
                    selections.width=frame2->width;
                selections.height=150;
                if (selections.height > frame2->height)
                    selections.height=frame2->height;
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
        cvAbsDiff(edge1,static_bg,edge1);
        //cvShowImage("ABDDiff",edge1);
        cvCanny(edge1,edge1,50.0,80.0,3);
        cvShowImage("Befor Not Edge",edge1);
        cvNot(edge1,edge1);
        cvDistTransform(edge1,edge_32f,2,3,0,0);
        cvMinMaxLoc(edge_32f,&min,&max,&pt1,&pt2,0);
        cvConvertScale(edge_32f,edge_32f,1.0/max,0);
        cvShowImage("Dist Transform",edge_32f);

        skin_score=GetSkinScore(frame2,skin_hist,non_skin_hist);
        //cvShowImage("skin_score_w/o_face",skin_score);
        cvSetImageROI(skin_score,selections);
        cvAndS(skin_score,cvScalarAll(0),skin_score,0);
        cvResetImageROI(skin_score);
        cvShowImage("skin_score_w/face",skin_score);
        cvMul(skin_score,bg,skin_score,1);
        cvMul(skin_score,imgframediff,skin_result,1);



        cvShowImage("back_subtraction",bg);
        cvMinMaxLoc(imgframediff,&min,&max,&pt1,&pt2,0);
        //printf("The max value is %f\n",max);
        if (max < 0.1)
        {
            cvCopy(imgframediff2,bgsub);
            cvCvtScale(bgsub,static_bg,1,0);
            cvRectangle(frame_temp_resize,pt2,cvPoint(pt2.x+20,pt2.y+20), CV_RGB(230,20,232), 3, 8, 0 );
            cvShowImage("name",frame_temp_resize);
        }
        else
        {
            CvScalar s;
            s.val[0]=0;
            frame_no=frame_no+1;
            for (k=0;k<detections;k++)
            {
                cvMinMaxLoc(skin_result,&min,&max,&min_point[k],&max_point[k],0);
                cvSet2D(skin_result,max_point[k].y,max_point[k].x,s);
                //printf("The Positions of skin detected for frame %d are x: %d y: %d %f\n",frame_no,max_point[k].x,max_point[k].y,max);
                /*pt1.x=max_point[k].x-15; pt1.y=max_point[k].y-15;
            if (pt1.x<0)
            pt1.x=0;
            if (pt1.y<0)
            pt1.y=0;
            pt2.x=max_point[k].x+15; pt2.y=max_point[k].y+15;
            if (pt2.x>frame2->width)
            pt2.x=frame2->width;
            if (pt2.y>frame2->height)
                pt2.y=frame2->height;*/
                if (k==0)
                {
                    start=clock();
                    feature=get_chamfer_dist(edge_32f,shape_template,cvPoint(max_point[0].x,max_point[0].y));
                    pt1.x=feature.x-15; pt1.y=feature.y-15;
                    if (pt1.x<0)
                        pt1.x=0;
                    if (pt1.y<0)
                        pt1.y=0;
                    pt2.x=feature.x+15; pt2.y=feature.y+15;
                    if (pt2.x>frame2->width)
                        pt2.x=frame2->width;
                    if (pt2.y>frame2->height)
                        pt2.y=frame2->height;
//                    printf(" The time of Execution was %f \n",((double) clock() - start) / CLOCKS_PER_SEC);
                    //cvRectangle( frame_temp_resize, pt1,pt2, CV_RGB(230,20,232), 3, 8, 0 );
                    cvShowImage("name",frame_temp_resize);
                }
            }


            /*	if ((frame_no >=8) && (frame_no <=13))
            {
                writetofile(edge1,edge_32f,imgframediff,bg,frame_temp,frame_temp_resize,frame_no);
            }*/
            //	start=clock();

            //printf(" The time of Execution was %f \n",((double) clock() - start) / CLOCKS_PER_SEC);
            //cvPoint(max_point[0].x-25,max_point[0].y-25)
            //printf("The distance value is %f\n",chamfer_dist);
            //cvRectangle( frame_temp_resize, cvPoint(max_point[0].x-37,max_point[0].y-37), cvPoint(max_point[0].x+38,max_point[0].y+38), CV_RGB(230,20,232), 3, 8, 0 );
            //cvShowImage("Chamfer Value",frame_temp_resize);
            //	}

            gesture_spotting(frame_no,feature); // Gesture Spotting Algorithm
        }
        if( cvWaitKey(10) >= 0)
        {
            //cvDestroyWindow("name");
            //	cvDestroyWindow("skin_score");
            cvReleaseImage(&frame1);
            cvReleaseImage(&frame2);
            cvReleaseImage(&imgframediff);
            cvReleaseImage(&imgframediff1);
            cvReleaseImage(&imgframediff2);
            cvReleaseImage(&skin_score);
            cvReleaseImage(&skin_result);
            cvReleaseCapture(&capture);
            break;
        }
        if (face_flag == 20)
        {cvCopy(imgframediff2,bgsub);
            shape_flag=0;}
        //}
        cvCopy(frame3,frame1);
        //value_check(imgframediff);
        //}
    }
    return 0;
}
