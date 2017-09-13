#include <iostream>
#include "readBMP.h"
#include <math.h>
#include <stdlib.h>
#include <boost/thread.hpp>
#include <GL/glut.h>
#include <fftw3.h>
#include <fstream>
#include <iterator>
#include <vector>
#include "sep_reader.h"

#define TRAIN

//SEPReader seismic_reader;//("/media/antonk/FreeAgent Drive/OpendTectData/Data/oxy/oxy.hdr");
//SEPReader   fault_reader;//("/home/antonk/SmartAFI/git/SmartAFI/out_pick");

//long o_x = seismic_reader.o3;
//long o_y = seismic_reader.o2;
//long o_z = seismic_reader.o1;

//long n_x = seismic_reader.n3;
//long n_y = seismic_reader.n2;
//long n_z = seismic_reader.n1;

//float * seismic_arr = new float[n_x*n_y*n_z];
//float *   fault_arr = new float[n_x*n_y*n_z];

long n_samples = 0;
long n_batch = 0;
long batch_iter = 20;
long n_variables = 0;
long n_cd = 1;
float epsilon = 0.001;

std::vector<float> errs;

long WIN=32;

float * img_arr = NULL;
float * orig_arr = NULL;

float * vis_preview = new float[WIN*WIN];
float * vis0_preview = new float[WIN*WIN];
float * vis1_preview = new float[WIN*WIN];
float * vis_previewG = new float[WIN*WIN];
float * vis0_previewG = new float[WIN*WIN];
float * vis1_previewG = new float[WIN*WIN];
float * vis_previewB = new float[WIN*WIN];
float * vis0_previewB = new float[WIN*WIN];
float * vis1_previewB = new float[WIN*WIN];

void vis2hid_worker(const float * X,float * H,long h,long v,float * c,float * W,std::vector<long> const & vrtx)
{
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t];
    for(long j=0;j<h;j++)
    {
      H[k*h+j] = c[j]; 
      for(long i=0;i<v;i++)
      {
        H[k*h+j] += W[i*h+j] * X[k*v+i];
      }
      H[k*h+j] = 1.0f/(1.0f + exp(-H[k*h+j]));
    }
  }
}

void hid2vis_worker(const float * H,float * V,long h,long v,float * b,float * W,std::vector<long> const & vrtx)
{
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t];
    for(long i=0;i<v;i++)
    {
      V[k*v+i] = b[i]; 
      for(long j=0;j<h;j++)
      {
        V[k*v+i] += W[i*h+j] * H[k*h+j];
      }
      V[k*v+i] = 1.0f/(1.0f + exp(-V[k*v+i]));
    }
  }
}



struct RBM
{
  long h; // number hidden elements
  long v; // number visible elements
  long n; // number of samples
  float * c; // bias term for hidden state, R^h
  float * b; // bias term for visible state, R^v
  float * W; // weight matrix R^h*v
  float * X; // input data, binary [0,1], v*n
  RBM(long _h,long _v,float * _W,float * _b,float * _c,long _n,float * _X)
  {
    for(long k=0;k<100;k++)
      std::cout << _X[k] << "\t";
    std::cout << "\n";
    X = _X;
    h = _h;
    v = _v;
    n = _n;
    c = _c;
    b = _b;
    W = _W;
  }
  RBM(long _h,long _v,long _n,float* _X)
  {
    for(long k=0;k<100;k++)
      std::cout << _X[k] << "\t";
    std::cout << "\n";
    X = _X;
    h = _h;
    v = _v;
    n = _n;
    c = new float[h];
    b = new float[v];
    W = new float[h*v];
    constant(c,0.5f,h);
    constant(b,0.5f,v);
    constant(W,0.5f,v*h);
  }

  float norm(float * dat,long size)
  {
    float ret = 0;
    for(long i=0;i<size;i++)
    {
      ret += dat[i]*dat[i];
    }
    return sqrt(ret);
  }

  void zero(float * dat,long size)
  {
    for(long i=0;i<size;i++)
    {
      dat[i] = 0;
    }
  }

  void constant(float * dat,float val,long size)
  {
    for(long i=0;i<size;i++)
    {
      dat[i] = (-1+2*((rand()%10000)/10000.0f))*val;
    }
  }

  void add(float * A, float * dA, float epsilon, long size)
  {
    for(long i=0;i<size;i++)
    {
      A[i] += epsilon * dA[i];
    }
  }

  void cd(long nGS,float epsilon,int offset=0)
  {

    std::cout << "cd" << std::endl;

    // CD Contrastive divergence (Hlongon's CD(k))
    //   [dW, db, dc, act] = cd(self, X) returns the gradients of
    //   the weihgts, visible and hidden biases using Hlongon's
    //   approximated CD. The sum of the average hidden units
    //   activity is returned in act as well.
   
    float * vis0 = new float[n*v];
    float * hid0 = new float[n*h];
    float * vis = new float[n*v];
    float * hid = new float[n*h];
    float * dW = new float[h*v];
    float * dc = new float[h];
    float * db = new float[v];

    for(long i=0,size=n*v;i<size;i++)
    {
      vis0[i] = X[i+offset];
    }
    /*
    float S;
    long N = 1;
    for(long k=0;k<n;k++)
    {
      float max_S = 1e-5;
      for(long y=0,i=0;y<WIN;y++)
      {
        for(long x=0;x<WIN;x++,i++)
        {
          if(i+N<WIN*WIN && x+N<WIN)
          {
            S = (pow(
                  vis0[2*WIN*WIN*k+i]
                 +vis0[2*WIN*WIN*k+i+1]
                 ,2))/((N+1)*(
                  pow(vis0[2*WIN*WIN*k+i],2)
                 +pow(vis0[2*WIN*WIN*k+i+1],2)
                  )+1e-5);
          }
          else
          {
            S = 1.0f;
          }
          //S *= S;
          //S *= S;
          vis0[2*WIN*WIN*k+i] = 1.0f - S;
          vis0[2*WIN*WIN*k+i] = 1.0f - S;
          vis0[2*WIN*WIN*k+i] = 1.0f - S;
          if(vis0[2*WIN*WIN*k+i]<0)vis0[2*WIN*WIN*k+i]=0;
          if(vis0[2*WIN*WIN*k+i]>1)vis0[2*WIN*WIN*k+i]=1;
          //vis0[2*WIN*WIN*k+i] = vis0[2*WIN*WIN*k+i]>0.5f;
          if(vis0[2*WIN*WIN*k+i]>max_S)max_S=vis0[2*WIN*WIN*k+i];
        }
      }
      for(long y=0,i=0;y<WIN;y++)
      {
        for(long x=0;x<WIN;x++,i++)
        {
          vis0[2*WIN*WIN*k+i] /= max_S;
          if(vis0[2*WIN*WIN*k+i]>1)vis0[2*WIN*WIN*k+i]=1;
        }
      }
    }
    */

    //for(long k=0;k<n;k++)
    //{
    //  float max_S = 1e-5;
    //  float min_S = 1;
    //  for(long y=0,i=0;y<WIN;y++)
    //  {
    //    for(long x=0;x<WIN;x++,i++)
    //    {
    //      if(vis0[2*WIN*WIN*k+i]>max_S)max_S=vis0[2*WIN*WIN*k+i];
    //      if(vis0[2*WIN*WIN*k+i]<min_S)min_S=vis0[2*WIN*WIN*k+i];
    //    }
    //  }
    //  for(long y=0,i=0;y<WIN;y++)
    //  {
    //    for(long x=0;x<WIN;x++,i++)
    //    {
    //      vis0[2*WIN*WIN*k+i] = (vis0[2*WIN*WIN*k+i]-min_S)/(max_S-min_S);
    //      if(vis0[2*WIN*WIN*k+i]<0)vis0[2*WIN*WIN*k+i]=0;
    //      if(vis0[2*WIN*WIN*k+i]>1)vis0[2*WIN*WIN*k+i]=1;
    //      vis0[2*WIN*WIN*k+i] -= 0.5f;
    //    }
    //  }
    //}

    vis2hid(vis0,hid0);

    for(long i=0;i<n*h;i++)
    {
      hid[i] = hid0[i];
    }

    for (long iter = 1;iter<=nGS;iter++)
    {
      std::cout << "iter=" << iter << std::endl;
      // sampling
      hid2vis(hid,vis);
      vis2hid(vis,hid);

      long off = rand()%(n);
      long offv = off*v;
      for(long x=0,k=0;x<WIN;x++)
      {
        for(long y=0;y<WIN;y++,k++)
        {
          vis_preview[k] = vis[offv+k];
          vis_previewG[k] = vis[offv+k+WIN*WIN];
          vis_previewB[k] = vis[offv+k+2*WIN*WIN];
        }
      }
      for(long x=0,k=0;x<WIN;x++)
      {
        for(long y=0;y<WIN;y++,k++)
        {
          vis1_preview[k] = orig_arr[offset+offv+k];
          vis1_previewG[k] = orig_arr[offset+offv+k+WIN*WIN];
          vis1_previewB[k] = orig_arr[offset+offv+k+2*WIN*WIN];
        }
      }
      long offh = off*h;
      for(long x=0,k=0;x<WIN;x++)
      {
        for(long y=0;y<WIN;y++,k++)
        {
          vis0_preview[k] = vis0[offh+k];
          vis0_previewG[k] = vis0[offh+k+WIN*WIN];
          vis0_previewB[k] = vis0[offh+k+2*WIN*WIN];
        }
      }

    }
  
    zero(dW,v*h);
    zero(dc,h);
    zero(db,v);
    float err = 0;
    for(long k=0;k<n;k++)
    {
      for(long i=0;i<v;i++)
      {
        for(long j=0;j<h;j++)
        {
          dW[i*h+j] -= (vis0[k*v+i]*hid0[k*h+j] - vis[k*v+i]*hid[k*h+j]) / n;
        }
      }

      for(long j=0;j<h;j++)
      {
        dc[j] -= (hid0[k*h+j]*hid0[k*h+j] - hid[k*h+j]*hid[k*h+j]) / n;
      }

      for(long i=0;i<v;i++)
      {
        db[i] -= (vis0[k*v+i]*vis0[k*v+i] - vis[k*v+i]*vis[k*v+i]) / n;
      }

      for(long i=0;i<v;i++)
      {
        err += (vis0[k*v+i]-vis[k*v+i])*(vis0[k*v+i]-vis[k*v+i]);
      }
    }
    err = sqrt(err);
    for(int t=2;t<10&&t<errs.size();t++)
      err += (errs[errs.size()+1-t]-err)/t;
    errs.push_back(err);
    add(W,dW,-epsilon,v*h);
    add(c,dc,-epsilon,h);
    add(b,db,-epsilon,v);

    std::cout << "dW norm = " << norm(dW,v*h) << std::endl;
    std::cout << "dc norm = " << norm(dc,h) << std::endl;
    std::cout << "db norm = " << norm(db,v) << std::endl;
    std::cout << "err = " << err << std::endl;

    delete [] vis0;
    delete [] hid0;
    delete [] vis;
    delete [] hid;
    delete [] dW;
    delete [] dc;
    delete [] db;

  }

  void sigmoid(float * p,float * X,long n)
  {
    for(long i=0;i<n;i++)
    {
      p[i] = 1.0f/(1.0f + exp(-X[i]));
    }
  }

  void vis2hid(const float * X,float * H)
  {
    std::vector<boost::thread*> threads;
    std::vector<std::vector<long> > vrtx(8);
    for(long i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(vis2hid_worker,X,H,h,v,c,W,vrtx[thread]));
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
  }
  
  void hid2vis(const float * H,float * V)
  {
    std::vector<boost::thread*> threads;
    std::vector<std::vector<long> > vrtx(8);
    for(long i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(hid2vis_worker,H,V,h,v,b,W,vrtx[thread]));
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
  }

};

struct DataUnit
{
  DataUnit *   hidden;
  DataUnit *  visible;
  DataUnit * visible0;
  long h,v;
  float * W;
  float * b;
  float * c;
  RBM * rbm;
  DataUnit(long _v,long _h)
  {
    v = _v;
    h = _h;
    W = new float[v*h];
    b = new float[v];
    c = new float[h];
      hidden = NULL;
     visible = NULL;
    visible0 = NULL;
  }

  void train(float * dat, long n,long num_iters)
  {
    rbm = new RBM(h,v,W,b,c,n,dat);
    for(long i=0;i<num_iters;i++)
    {
      rbm->cd(10,.0000001);
    }
  }

  void transform(float* X,float* Y)
  {
    rbm->vis2hid(X,Y);
  }
};

struct CurveletTransform
{
  void operator () (long nx,long ny,float * dat,float * muted,float target_th = 0,float cut_off = 0)
  {
    fftwf_complex * in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*nx*ny);
    for(long x=0,k=0;x<nx;x++)
      for(long y=0;y<ny;y++,k++)
      {
        in[k][0] = dat[k];
        in[k][1] = 0;
      }
    fftwf_complex * out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*nx*ny);
    fftwf_complex * ret = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*nx*ny);
    fftwf_plan plan_forward  = fftwf_plan_dft_2d(nx,ny,in ,out,FFTW_FORWARD ,FFTW_ESTIMATE); 
    fftwf_plan plan_backward = fftwf_plan_dft_2d(nx,ny,out,ret,FFTW_BACKWARD,FFTW_ESTIMATE); 
    fftwf_execute(plan_forward);
    float re,im,th,X,Y,R,D,TH=1;
    float S = 5;
    float G = 0.5;
    for(long x=0,k=0;x<nx;x++)
      for(long y=0;y<ny;y++,k++)
      {
        X = x;
        if(X>nx/2)X-=nx;
        Y = y;
        if(Y>ny/2)Y-=ny;
        D = X*X+Y*Y;
        //if(4*D < nx*nx)
        {
          R = exp(-D*G);
          //th = atan2(Y,X) - target_th;
          //TH = exp(-th*th*S)+exp(-(th-M_PI)*(th-M_PI)*S);
          //TH *= sin(th);
          //TH = exp(-S*X*X);
          re = out[k][0];
          im = out[k][1];
          out[k][0] = re*R*TH;
          out[k][1] = im*R*TH;
        }
        //else
        //{
        //  out[k][0] = 0;
        //  out[k][1] = 0;
        //}
      }
    fftwf_execute(plan_backward);
    fftwf_destroy_plan(plan_forward);
    fftwf_destroy_plan(plan_backward);
    for(long x=0,k=0;x<nx;x++)
      for(long y=0;y<ny;y++,k++)
      {
        muted[k] = 30*ret[k][0]/(nx*ny);
      }
    fftwf_free(in);
    fftwf_free(out);
    fftwf_free(ret);
  }
};

struct FeatureDetectorLayer
{
  void detect_features (long nx,long ny,float * data,float * features)
  {
    // Curvelet Transform
    CurveletTransform transform;
    for(float cut_off=0;cut_off<5;cut_off++)
    {
      for(float th=0;th<M_PI;th+=M_PI/16)
      {
        //transform(nx,ny,data,features,th,cut_off);
      }
    }


  }
  void pooling(long nx,long ny,long poolx,long pooly,float * data,float * features)
  {
    // max ( ... )
    for(long x=0,dx=0,k=0;x<nx;x+=poolx,dx++)
    {
      for(long y=0,dy=0;y<ny;y+=pooly,dy++,k++)
      {
        float max_val = 0;
        for(long wx=0;wx<poolx;wx++)
        {
          for(long wy=0;wy<pooly;wy++)
          {
            //max_val = max(max_val,data[(x+wx)*ny+(y+wy)]);
            features[k] = max_val;
          }
        }
      }
    }
  }
  void ReLU(long n,float * dat)
  {
    // softmax
    // f(x) = x^+ = max(0,x)
    // f(x) ~ ln(1+exp(x))
    // basically logistic sigmoid function if you want to use real math
    // f(x) = 1/(1+exp(-x))
    for(long k=0;k<n;k++)
      dat[k] = 1/(1+exp(-dat[k]));
  }
};

// Multi Layer RBM
//
//  Auto-encoder
//
//          [***]
//         /     \
//     [*****] [*****]
//       /         \
// [********]   [********]
//   inputs      outputs
//
struct mRBM
{
  std::vector<DataUnit*>  input_branch;
  std::vector<DataUnit*> output_branch;
  DataUnit* bottle_neck;
  void addInputDatUnit(long v,long h)
  {
    DataUnit * unit = new DataUnit(v,h);
    input_branch.push_back(unit);
  }
  void addOutputDatUnit(long v,long h)
  {
    output_branch.push_back(new DataUnit(v,h));
  }
  void addBottleNeckDatUnit(long v,long h)
  {
    bottle_neck = new DataUnit(v,h);
  }
  void construct(std::vector<long> input_num,std::vector<long> output_num,long bottle_neck_num)
  {
    for(long i=0;i+1<input_num.size();i++)
    {
      input_branch.push_back(new DataUnit(input_num[i],input_num[i+1]));
    }
    for(long i=0;i+1<output_num.size();i++)
    {
      output_branch.push_back(new DataUnit(output_num[i],output_num[i+1]));
    }
    bottle_neck = new DataUnit(input_num[input_num.size()-1]+output_num[output_num.size()-1],bottle_neck_num);
  }
  mRBM()
  {

  }
  void copy(float * X,float * Y,long num)
  {
    for(long i=0;i<num;i++)
    {
      Y[i] = X[i];
    }
  }
  void train(long in_num,long out_num,long n_samp,float * in,float * out)
  {
    long n_iter = 100;
    float * X = NULL;
    float * Y = NULL;
    X = new float[in_num*n_samp];
    for(long i=0;i<input_branch.size();i++)
    {
      //if(i>0)input_branch[i]->initialize_weights(input_branch[i-1]); // initialize weights to transpose of previous layer weights M_i -> W = M_{i-1} -> W ^ T
      input_branch[i]->train(X,n_samp,n_iter);
      Y = new float[input_branch[i]->h*n_samp];
      input_branch[i]->transform(X,Y);
      delete [] X;
      X = NULL;
      X = new float[input_branch[i]->h*n_samp];
      copy(Y,X,output_branch[i]->h*n_samp);
      delete [] Y;
      Y = NULL;
    }
    delete [] X;
    X = NULL;
    X = new float[out_num*n_samp];
    for(long i=0;i<output_branch.size();i++)
    {
      //if(i>0)output_branch[i]->initialize_weights(output_branch[i-1]); // initialize weights to transpose of previous layer weights M_i -> W = M_{i-1} -> W ^ T
      output_branch[i]->train(X,n_samp,n_iter);
      Y = new float[output_branch[i]->h*n_samp];
      output_branch[i]->transform(X,Y);
      delete [] X;
      X = NULL;
      X = new float[output_branch[i]->h*n_samp];
      copy(Y,X,output_branch[i]->h*n_samp);
      delete [] Y;
      Y = NULL;
    }
    delete [] X;
    X = NULL;
    X = new float[(input_branch[input_branch.size()-1]->h + output_branch[output_branch.size()-1]->h)*n_samp];
    bottle_neck->train(X,n_samp,n_iter);
    delete [] X;
    X = NULL;
  }
};

RBM * rbm = NULL;

void drawBox(void)
{
  //std::cout << "drawBox" << std::endl;
  float max_err = 0;
  for(long k=0;k<errs.size();k++)
  {
    if(max_err<errs[k])max_err=errs[k];
  }
  glColor3f(1,1,1);
  glBegin(GL_LINES);
  for(long k=0;k+1<errs.size();k++)
  {
    glVertex3f( -1 + 2*k / ((float)errs.size()-1)
              , errs[k] / max_err
              , 0
              );
    glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
              , errs[k+1] / max_err
              , 0
              );
    glVertex3f( -1 + 2*k / ((float)errs.size()-1)
              , 0
              , 0
              );
    glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
              , 0
              , 0
              );
    glVertex3f( -1 + 2*k / ((float)errs.size()-1)
              , 0
              , 0
              );
    glVertex3f( -1 + 2*k / ((float)errs.size()-1)
              , errs[k] / max_err
              , 0
              );
    glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
              , 0
              , 0
              );
    glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
              , errs[k+1] / max_err
              , 0
              );
  }
  glEnd();
  if(rbm)
  {
    float max_W = -1000;
    float min_W =  1000;
    for(long i=0,k=0;i<rbm->v;i++)
      for(long j=0;j<rbm->h;j++,k++)
      {
        if(rbm->W[k]>max_W)max_W=rbm->W[k];
        if(rbm->W[k]<min_W)min_W=rbm->W[k];
      }
    float fact_W = 1.0 / (max_W - min_W);
    float col;
    glBegin(GL_QUADS);
    float d=3e-3;
    for(long x=0;x<WIN;x++)
    {
      for(long y=0;y<WIN;y++)
      {
        for(long i=0;i<rbm->v/WIN;i++)
        {
          for(long j=0;j<rbm->h/WIN;j++)
          {
            col = 0.5f + 0.5f*(rbm->W[(i+x)*rbm->h+j+y]-min_W)*fact_W;
            glColor3f(col,col,col);
            glVertex3f(  -1+(1.15*WIN*i+x)/(1.15*(float)rbm->v) ,  -1+(1.15*WIN*j+y)/(1.15*(float)rbm->h),0);
            glVertex3f(d+-1+(1.15*WIN*i+x)/(1.15*(float)rbm->v) ,  -1+(1.15*WIN*j+y)/(1.15*(float)rbm->h),0);
            glVertex3f(d+-1+(1.15*WIN*i+x)/(1.15*(float)rbm->v) ,d+-1+(1.15*WIN*j+y)/(1.15*(float)rbm->h),0);
            glVertex3f(  -1+(1.15*WIN*i+x)/(1.15*(float)rbm->v) ,d+-1+(1.15*WIN*j+y)/(1.15*(float)rbm->h),0);
          }
        }
      }
    }
    glEnd();
  }
  {
    float d = 5e-1;
    float col;
    glBegin(GL_QUADS);
    for(long y=0,k=0;y<WIN;y++)
    {
      for(long x=0;x<WIN;x++,k++)
      {
        glColor3f(vis_preview[k]
                 ,vis_previewG[k]
                 ,vis_previewB[k]
                 );
        glVertex3f(      (x)/(2.0*WIN) ,      -1+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(d/WIN+(x)/(2.0*WIN) ,      -1+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(d/WIN+(x)/(2.0*WIN) ,d/WIN+-1+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(      (x)/(2.0*WIN) ,d/WIN+-1+(2*WIN-1-y)/(2.0*WIN),0);
      }
    }
    for(long y=0,k=0;y<WIN;y++)
    {
      for(long x=0;x<WIN;x++,k++)
      {
        glColor3f(vis0_preview[k]
                 ,vis0_previewG[k]
                 ,vis0_previewB[k]
                 );
        glVertex3f(      0.5f+(x)/(2.0*WIN) ,      -1+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(d/WIN+0.5f+(x)/(2.0*WIN) ,      -1+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(d/WIN+0.5f+(x)/(2.0*WIN) ,d/WIN+-1+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(      0.5f+(x)/(2.0*WIN) ,d/WIN+-1+(2*WIN-1-y)/(2.0*WIN),0);
      }
    }
    for(long y=0,k=0;y<WIN;y++)
    {
      for(long x=0;x<WIN;x++,k++)
      {
        glColor3f(vis1_preview[k]
                 ,vis1_previewG[k]
                 ,vis1_previewB[k]
                 );
        glVertex3f(      0.5f+(x)/(2.0*WIN) ,      -1-0.5f+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(d/WIN+0.5f+(x)/(2.0*WIN) ,      -1-0.5f+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(d/WIN+0.5f+(x)/(2.0*WIN) ,d/WIN+-1-0.5f+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(      0.5f+(x)/(2.0*WIN) ,d/WIN+-1-0.5f+(2*WIN-1-y)/(2.0*WIN),0);
      }
    }
    glEnd();
  }
}

void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  drawBox();
  glutSwapBuffers();
}

void idle(void)
{
  usleep(10000);
  glutPostRedisplay();
}

void init(void)
{
  /* Use depth buffering for hidden surface elimination. */
  glEnable(GL_DEPTH_TEST);

  /* Setup the view of the cube. */
  glMatrixMode(GL_PROJECTION);
  gluPerspective( /* field of view in degree */ 40.0,
    /* aspect ratio */ 1.0,
    /* Z near */ 1.0, /* Z far */ 10.0);
  glMatrixMode(GL_MODELVIEW);
  gluLookAt(0.0, 0.0, 3,  /* eye is at (0,0,5) */
    0.0, 0.0, 0.0,      /* center is at (0,0,0) */
    0.0, 1.0, 0.);      /* up is in positive Y direction */
}

void run_rbm(long v,long h,long n,long total_n,float * dat)
{
  rbm = new RBM(v,h,n,dat);
  for(long i=0;;i++)
  {
    std::cout << "i=" << i << std::endl;
    long offset = (rand()%(total_n-n));
    for(long k=0;k<batch_iter;k++)
    {
      std::cout << "prog:" << 100*(float)k/batch_iter << "%" << std::endl;
      rbm->cd(n_cd,epsilon,offset*(10+3*WIN*WIN));
    }
  }
}

struct fault
{
  bool init;
  bool preview;
  float cx,cy; // fault center
  float vx,vy; // fault orientation
  float sx,sy; // structure orientation
  long wx,wy; // window width
  std::vector<float> amp; // seismic event amplitudes
  std::vector<float> shift; // seismic event shifts
  float fault_shift; // fault shift
  void randomize()
  {
    long num_events = 10+rand()%10;
    amp = std::vector<float>(num_events);
    shift = std::vector<float>(num_events);
    for(long i=0;i<num_events;i++)
    {
      amp[i] = (-1+2*(rand()%10000)/10000.0f);
      shift[i] = WIN*(-1+2*(rand()%10000)/10000.0f);
    }
    cx = 0.25+0.5*(rand()%10000)/10000.0f;
    cy = 0.25+0.5*(rand()%10000)/10000.0f;
    fault_shift = (rand()%10>=5)?4:-4;//10*(-1+2*(rand()%10000)/10000.0f);
    vx = 1;
    vy = 0.2f*(-1+2*(rand()%10000)/10000.0f);
    sx = 0;//0.2f*(-1+2*(rand()%10000)/10000.0f);
    sy = 1;
    init = true;
    preview = false;
  }
  void generate_data(long _wx,long _wy,float * dat)
  {
    if(init)
    {
      wx = _wx;
      wy = _wy;
      for(long i=0;i<2*wx*wy;i++)
      {
        dat[i] = 0;
      }
      long off = wx*wy;
      // generate data based on parameters
      float dx,dy;
      float vdot,sdot;
      for(long n=0;n<amp.size();n++)
      {
        for(long y=0,i=0;y<wy;y++)
        {
          for(long x=0;x<wx;x++,i++)
          {
            vdot = vx*(x-wx*cx) + vy*(y-wy*cy);
            sdot = sx*(x-wx*cx) + sy*(y-wy*cy);
            if(vdot>0)
            {
              dx = fault_shift + shift[n] - sdot;
              //dat[i] += amp[n]*exp(-dx*dx*WIN);
              dat[i] += amp[n]*sin(dx*2) + 0.05f*(rand()%10000)/10000.0f;
            }
            else
            {
              dx = shift[n] - sdot;
              //dat[i] += amp[n]*exp(-dx*dx*WIN);
              dat[i] += amp[n]*sin(dx*2) + 0.05f*(rand()%10000)/10000.0f;
            }
            dy = vdot;
            dat[i+off] += fabs(fault_shift) * exp(-dy*dy*10);
          }
        }
      }

      for(long y=0,i=0;y<wy;y++)
      {
        for(long x=0;x<wx;x++,i++)
        {
          //dat[i] = 0.5f + 0.15f*dat[i];
          dat[i+off] = dat[i+off]>0.5f?1:0;
        }
      }

      if(false)
      {
        std::cout << "$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
        for(long x=0,i=0;x<wx;x++)
        {
          for(long y=0;y<wy;y++,i++)
          {
            std::cout << dat[i];
          }
          std::cout << std::endl;
        }
        std::cout << "=======================" << std::endl;
        for(long x=0,i=0;x<wx;x++)
        {
          for(long y=0;y<wy;y++,i++)
          {
            std::cout << dat[i+off];
          }
          std::cout << std::endl;
        }
      }

    }
    else
    {
      std::cout << "fault not initialized" << std::endl;
      exit(1);
    }
  }
  fault()
  {
    init = false;
    randomize();
  }
};

/*
std::vector<long> indices;

struct real_fault
{
  bool init;
  long ox,oy,oz;
  long nx,ny,nz;
  void randomize()
  {
    init = true;
    while(true)
    {
      long id = indices[rand()%indices.size()];
      ox = (id/(n_z*n_y))%n_x-nx/2;
      oy = (id/n_z)%n_y-ny/2;
      oz = id%n_z-nz/2;
      float max_arr = 0;
      float val;
      for(long i=0,x=ox+nx/4;x+nx/4<ox+nx;x++)
      for(long y=oy+ny/4;y+ny/4<oy+ny;y++)
      for(long z=oz+nz/4;z+nz/4<oz+nz;z++,i++)
      {
        val = fault_arr[z+n_z*(y+n_y*x)];
        if(val>max_arr)max_arr=val;
      }
      std::cout << "max_arr=" << max_arr << std::endl;
      if(max_arr>0.9){break;}
    }
  }
  void generate_data(float * dat)
  {
    if(init)
    {
      long off = nx*ny*nz;
      for(long i=0,z=oz;z<oz+nz;z++)
      for(long y=oy;y<oy+ny;y++)
      for(long x=ox;x<ox+nx;x++,i++)
      {
        dat[i    ] = seismic_arr[z+n_z*(y+n_y*x)];
        dat[i+off] =   fault_arr[z+n_z*(y+n_y*x)];
      }
    }
    else
    {
      std::cout << "fault not initialized" << std::endl;
      exit(1);
    }
  }
  real_fault(long _nx,long _ny,long _nz)
  {
    nx = _nx;
    ny = _ny;
    nz = _nz;
    init = false;
    randomize();
  }
};
*/

/*
void faultGenerator2D(long wx,long wy,long n,float* dat)
{
  for(long k=0;k<n;k++)
  {
    // generate sample
    fault f;
    f.generate_data(wx,wy,&dat[k*2*wx*wy]);
  }
}
*/

/*
void realFaultGenerator2D(long wx,long wy,long n,float * dat)
{
  std::cout << "real fault generator" << std::endl;
	seismic_reader.read_sepval  ( &seismic_arr[0]
		                          , seismic_reader.o1
		                          , seismic_reader.o2
		                          , seismic_reader.o3
		                          , seismic_reader.n1
		                          , seismic_reader.n2
		                          , seismic_reader.n3
		                          );
  float max_seismic = 0;
  for(long i=0,size=seismic_reader.n1*seismic_reader.n2*seismic_reader.n3;i<size;i++)
  {
    if(max_seismic<seismic_arr[i])max_seismic=seismic_arr[i];
  }
  for(long i=0,size=seismic_reader.n1*seismic_reader.n2*seismic_reader.n3;i<size;i++)
  {
    seismic_arr[i] /= 1e-5 + max_seismic;
  }
  std::cout << "seismic reader done" << std::endl;
	  fault_reader.read_sepval  ( &  fault_arr[0]
		                          , fault_reader.o1
		                          , fault_reader.o2
		                          , fault_reader.o3
		                          , fault_reader.n1
		                          , fault_reader.n2
		                          , fault_reader.n3
		                          );
  float max_fault = 0;
  float min_fault = 1;
  for(long i=0,size=fault_reader.n1*fault_reader.n2*fault_reader.n3;i<size;i++)
  {
    if(max_fault<fault_arr[i])max_fault=fault_arr[i];
    if(min_fault>fault_arr[i])min_fault=fault_arr[i];
  }
  std::cout << "max-fault:" << max_fault << std::endl;
  std::cout << "min-fault:" << min_fault << std::endl;
  std::cout << "fault reader done" << std::endl;
  for(long k=0,x=0;x<n_x;x++)
    for(long y=0;y<n_y;y++)
      for(long z=0;z<n_z;z++,k++)
      {
        if(x>WIN)
        if(y>WIN)
        if(z>WIN)
        if(x<n_x-WIN)
        if(y<n_y-WIN)
        if(z<n_z-WIN)
        if(fault_arr[k]>0)
          indices.push_back(k);
      }
  for(long k=0;k<n;k++)
  {
    // generate sample
    real_fault f(wx,1,wy);
    f.generate_data(&dat[k*2*wx*wy]);
  }
  std::cout << "generating data done" << std::endl;
}
*/

void keyboard(unsigned char Key, int x, int y)
{
  switch(Key)
  {
    case ' ':
      {
        long v = 3*WIN*WIN+10;
        long h = 3*WIN*WIN+10;
        long n = n_samples;
        boost::thread * thr ( new boost::thread(run_rbm,v,h,n_batch,n,img_arr) );
        break;
      }
    case 27:
      {
        exit(1);
        break;
      }
  };
}

void readBinary(std::string filename, std::vector<char> & out)
{
  std::ifstream input( filename.c_str(), std::ios::binary );
  // copies all data longo buffer
  std::vector<char> tmp((
    std::istreambuf_iterator<char>(input)), 
    (std::istreambuf_iterator<char>()));
  for(int k=0;k<tmp.size();k++)
    out.push_back(tmp[k]);
}

struct SemblanceLayer
{

  float * tmp1_n;
  float * tmp2_n;
  float * tmp3_n;
  float * tmp4_n;
  float * tmp1_d;
  float * tmp2_d;
  float * tmp3_d;
  float * tmp4_d;

  SemblanceLayer(int nx,int ny)
  {
    tmp1_n = new float[3*nx*ny];
    tmp2_n = new float[3*nx*ny];
    tmp3_n = new float[3*nx*ny];
    tmp4_n = new float[3*nx*ny];
    tmp1_d = new float[3*nx*ny];
    tmp2_d = new float[3*nx*ny];
    tmp3_d = new float[3*nx*ny];
    tmp4_d = new float[3*nx*ny];
  }

  void operator() (int nx,int ny,float * in,float * out)
  {
    // calculate semblance
    //
    //    . . .
    //    + + +
    //    . . .
    //
    memset(tmp1_n,0,sizeof(float)*3*nx*ny);
    memset(tmp1_d,0,sizeof(float)*3*nx*ny);
    for(int c=0;c<3;c++)
    for(int x=1;x+1<nx;x++)
      for(int y=1;y+1<ny;y++)
      {
        tmp1_n[nx*ny*c+y+ny*x] += in[nx*ny*c+y+  ny*x];
        tmp1_n[nx*ny*c+y+ny*x] += in[nx*ny*c+y+1+ny*x];
        tmp1_n[nx*ny*c+y+ny*x] += in[nx*ny*c+y-1+ny*x];
        tmp1_n[nx*ny*c+y+ny*x] *= tmp1_n[nx*ny*c+y+ny*x];
        tmp1_d[nx*ny*c+y+ny*x] += in[nx*ny*c+y+  ny*x]*in[nx*ny*c+y+  ny*x];
        tmp1_d[nx*ny*c+y+ny*x] += in[nx*ny*c+y+1+ny*x]*in[nx*ny*c+y+1+ny*x];
        tmp1_d[nx*ny*c+y+ny*x] += in[nx*ny*c+y-1+ny*x]*in[nx*ny*c+y-1+ny*x];
        tmp1_d[nx*ny*c+y+ny*x] *= 3;
      }

    // calculate semblance
    //
    //    + . .
    //    . + .
    //    . . +
    //
    memset(tmp2_n,0,sizeof(float)*3*nx*ny);
    memset(tmp2_d,0,sizeof(float)*3*nx*ny);
    for(int c=0;c<3;c++)
    for(int x=1;x+1<nx;x++)
      for(int y=1;y+1<ny;y++)
      {
        tmp2_n[nx*ny*c+y+ny*x] += in[nx*ny*c+y+  ny*x];
        tmp2_n[nx*ny*c+y+ny*x] += in[nx*ny*c+y+1+ny*(x+1)];
        tmp2_n[nx*ny*c+y+ny*x] += in[nx*ny*c+y-1+ny*(x-1)];
        tmp2_n[nx*ny*c+y+ny*x] *= tmp2_n[nx*ny*c+y+ny*x];
        tmp2_d[nx*ny*c+y+ny*x] += in[nx*ny*c+y+  ny*x]*in[nx*ny*c+y+  ny*x];
        tmp2_d[nx*ny*c+y+ny*x] += in[nx*ny*c+y+1+ny*(x+1)]*in[nx*ny*c+y+1+ny*(x+1)];
        tmp2_d[nx*ny*c+y+ny*x] += in[nx*ny*c+y-1+ny*(x-1)]*in[nx*ny*c+y-1+ny*(x-1)];
        tmp2_d[nx*ny*c+y+ny*x] *= 3;
      }

    // calculate semblance
    //
    //    . + .
    //    . + .
    //    . + .
    //
    memset(tmp3_n,0,sizeof(float)*3*nx*ny);
    memset(tmp3_d,0,sizeof(float)*3*nx*ny);
    for(int c=0;c<3;c++)
    for(int x=1;x+1<nx;x++)
      for(int y=1;y+1<ny;y++)
      {
        tmp3_n[nx*ny*c+y+ny*x] += in[nx*ny*c+y+ny*x];
        tmp3_n[nx*ny*c+y+ny*x] += in[nx*ny*c+y+ny*(x+1)];
        tmp3_n[nx*ny*c+y+ny*x] += in[nx*ny*c+y+ny*(x-1)];
        tmp3_n[nx*ny*c+y+ny*x] *= tmp3_n[nx*ny*c+y+ny*x];
        tmp3_d[nx*ny*c+y+ny*x] += in[nx*ny*c+y+ny*x]*in[nx*ny*c+y+ny*x];
        tmp3_d[nx*ny*c+y+ny*x] += in[nx*ny*c+y+ny*(x+1)]*in[nx*ny*c+y+ny*(x+1)];
        tmp3_d[nx*ny*c+y+ny*x] += in[nx*ny*c+y+ny*(x-1)]*in[nx*ny*c+y+ny*(x-1)];
        tmp3_d[nx*ny*c+y+ny*x] *= 3;
      }

    // calculate semblance
    //
    //    . . +
    //    . + .
    //    + . .
    //
    memset(tmp4_n,0,sizeof(float)*3*nx*ny);
    memset(tmp4_d,0,sizeof(float)*3*nx*ny);
    for(int c=0;c<3;c++)
    for(int x=1;x+1<nx;x++)
      for(int y=1;y+1<ny;y++)
      {
        tmp4_n[nx*ny*c+y+ny*x] += in[nx*ny*c+y+  ny*x];
        tmp4_n[nx*ny*c+y+ny*x] += in[nx*ny*c+y+1+ny*(x-1)];
        tmp4_n[nx*ny*c+y+ny*x] += in[nx*ny*c+y-1+ny*(x+1)];
        tmp4_n[nx*ny*c+y+ny*x] *= tmp4_n[nx*ny*c+y+ny*x];
        tmp4_d[nx*ny*c+y+ny*x] += in[nx*ny*c+y+  ny*x]*in[nx*ny*c+y+  ny*x];
        tmp4_d[nx*ny*c+y+ny*x] += in[nx*ny*c+y+1+ny*(x-1)]*in[nx*ny*c+y+1+ny*(x-1)];
        tmp4_d[nx*ny*c+y+ny*x] += in[nx*ny*c+y-1+ny*(x+1)]*in[nx*ny*c+y-1+ny*(x+1)];
        tmp4_d[nx*ny*c+y+ny*x] *= 3;
      }

    memset(out,0,sizeof(float)*3*nx*ny);
    float semb,max_semb,tmp_semb;
    for(int c=0;c<3;c++)
    for(int x=0,X=0;x<nx;x+=2,X++)
      for(int y=0,Y=0;y<ny;y+=2,Y++)
      {
        
        max_semb = 0;
        semb = tmp1_n[nx*ny*c+y+ny*x] / (tmp1_d[nx*ny*c+y+ny*x]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        semb = tmp1_n[nx*ny*c+y+1+ny*x] / (tmp1_d[nx*ny*c+y+1+ny*x]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        semb = tmp1_n[nx*ny*c+y+ny*(x+1)] / (tmp1_d[nx*ny*c+y+ny*(x+1)]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        semb = tmp1_n[nx*ny*c+y+1+ny*(x+1)] / (tmp1_d[nx*ny*c+y+1+ny*(x+1)]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        out[nx*ny*c+Y+ny*X] = 1/(1+exp(-5.0*max_semb));
        
        
        max_semb = 0;
        semb = tmp2_n[nx*ny*c+y+ny*x] / (tmp2_d[nx*ny*c+y+ny*x]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        semb = tmp2_n[nx*ny*c+y+1+ny*x] / (tmp2_d[nx*ny*c+y+1+ny*x]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        semb = tmp2_n[nx*ny*c+y+ny*(x+1)] / (tmp2_d[nx*ny*c+y+ny*(x+1)]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        semb = tmp2_n[nx*ny*c+y+1+ny*(x+1)] / (tmp2_d[nx*ny*c+y+1+ny*(x+1)]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        out[nx*ny*c+Y+ny/2+ny*X] = 1/(1+exp(-5.0*max_semb));

        
        max_semb = 0;
        semb = tmp3_n[nx*ny*c+y+ny*x] / (tmp3_d[nx*ny*c+y+ny*x]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        semb = tmp3_n[nx*ny*c+y+1+ny*x] / (tmp3_d[nx*ny*c+y+1+ny*x]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        semb = tmp3_n[nx*ny*c+y+ny*(x+1)] / (tmp3_d[nx*ny*c+y+ny*(x+1)]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        semb = tmp3_n[nx*ny*c+y+1+ny*(x+1)] / (tmp3_d[nx*ny*c+y+1+ny*(x+1)]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        out[nx*ny*c+Y+ny*(X+nx/2)] = 1/(1+exp(-5.0*max_semb));


        max_semb = 0;
        semb = tmp4_n[nx*ny*c+y+ny*x] / (tmp4_d[nx*ny*c+y+ny*x]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        semb = tmp4_n[nx*ny*c+y+1+ny*x] / (tmp4_d[nx*ny*c+y+1+ny*x]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        semb = tmp4_n[nx*ny*c+y+ny*(x+1)] / (tmp4_d[nx*ny*c+y+ny*(x+1)]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        semb = tmp4_n[nx*ny*c+y+1+ny*(x+1)] / (tmp4_d[nx*ny*c+y+1+ny*(x+1)]+1e-5);
        semb *= semb;
        semb *= semb;
        tmp_semb = 1-semb;
        if(tmp_semb>max_semb)max_semb = tmp_semb;
        out[nx*ny*c+Y+ny/2+ny*(X+nx/2)] = 1/(1+exp(-5.0*max_semb));
        
      }
  }
};

int main(int argc,char ** argv)
{
  srand(time(0));

  std::vector<char> data;
  readBinary( "/home/antonk/cifar10/cifar10_data/cifar-10-batches-bin/data_batch_1.bin", data );
  readBinary( "/home/antonk/cifar10/cifar10_data/cifar-10-batches-bin/data_batch_2.bin", data );
  readBinary( "/home/antonk/cifar10/cifar10_data/cifar-10-batches-bin/data_batch_3.bin", data );
  readBinary( "/home/antonk/cifar10/cifar10_data/cifar-10-batches-bin/data_batch_4.bin", data );
  readBinary( "/home/antonk/cifar10/cifar10_data/cifar-10-batches-bin/data_batch_5.bin", data );
  readBinary( "/home/antonk/cifar10/cifar10_data/cifar-10-batches-bin/test_batch.bin", data );

  //long off1 = rand()%100;
  //long off2 = rand()%100;
  //for(long x=0,k=0;x<WIN;x++)
  //  for(long y=0;y<WIN;y++,k++)
  //  {
  //    vis0_preview [k] = (unsigned char)data[k+1+off1*(3*WIN*WIN+1)]/256.0;
  //    vis0_previewG[k] = (unsigned char)data[k+1+off1*(3*WIN*WIN+1)+(WIN*WIN)]/256.0;
  //    vis0_previewB[k] = (unsigned char)data[k+1+off1*(3*WIN*WIN+1)+2*(WIN*WIN)]/256.0;
  //     vis_preview [k] = (unsigned char)data[k+1+off2*(3*WIN*WIN+1)]/256.0;
  //     vis_previewG[k] = (unsigned char)data[k+1+off2*(3*WIN*WIN+1)+(WIN*WIN)]/256.0;
  //     vis_previewB[k] = (unsigned char)data[k+1+off2*(3*WIN*WIN+1)+2*(WIN*WIN)]/256.0;
  //  }

  n_samples = 50;//data.size() / (3*WIN*WIN+1);
  n_batch = n_samples - 1;

  // visible = 3*WIN*WIN
  // hidden = 3*WIN*WIN + 10
  n_variables = ((3*WIN*WIN + 10));

  img_arr = new float[n_samples * n_variables];
  orig_arr = new float[n_samples * n_variables];

  for(long s=0,k=0,t=0;s<n_samples;s++)
  {
    for(long i=0;i<3*WIN*WIN;i++,k++,t++)
    img_arr[k] = (unsigned char)data[1+(3*WIN*WIN+1)*s+i]/256.0;
    for(long i=0;i<10;i++,k++)
    img_arr[k] = (long)data[(3*WIN*WIN+1)*s]==k;
    t++;
  }
  for(int i=0;i<n_samples*n_variables;i++)
    orig_arr[i] = img_arr[i];
  SemblanceLayer semb_layer(WIN,WIN);
  for(long s=0;s<n_samples;s++)
  {
    semb_layer(WIN,WIN,&img_arr[(3*WIN*WIN+10)*s],&img_arr[(3*WIN*WIN+10)*s]);
  }

  std::cout << "done loading data ... " << std::endl;

  //CurveletTransform trans;
  //long nx = 32;
  //long ny = 32;
  //float dat[1024];
  //for(long x=0,k=0;x<nx;x++)
  //  for(long y=0;y<ny;y++,k++)
  //  {
  //    dat[k] = (x==nx/2&&y==ny/2)?1:0;
  //  }
  //float ret[1024];
  //trans(nx,ny,dat,ret,0);
  //for(long x=0,k=0;x<nx;x++)
  //  for(long y=0;y<ny;y++,k++)
  //  {
  //    vis0_preview [k] = dat[k];
  //    vis0_previewG[k] = dat[k];
  //    vis0_previewB[k] = dat[k];
  //     vis_preview [k] = ret[k];
  //     vis_previewG[k] = ret[k];
  //     vis_previewB[k] = ret[k];
  //  }
  std::cout << "Press space to start..." << std::endl;
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutCreateWindow("Boltzmann Machine");
  glutDisplayFunc(display);
  glutIdleFunc(idle);
  glutKeyboardFunc(keyboard);
  init();
  glutMainLoop();
  return 0;
}

