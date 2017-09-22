#include <iostream>
#include "readBMP.h"
#include <math.h>
#include <stdlib.h>
#include <boost/date_time/posix_time/posix_time.hpp>
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

long dat_offset = 0;

long n_samples = 0;
long n_batch = 0;
long batch_iter = 200;
long n_variables = 0;
long n_cd = 1;
float c_epsilon = 0.001;

std::vector<float> errs;

long WIN=32;

float * img_arr = NULL;
float * out_arr = NULL;
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

void clear() {
  // CSI[2J clears screen, CSI[H moves the cursor to top-left corner
  std::cout << "\x1B[2J\x1B[H";
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

struct gradient_info
{
  long n;
  long v;
  long h;
  float * vis0;
  float * hid0;
  float * vis;
  float * hid;
  float * dW;
  float * dc;
  float * db;
  float partial_err;
  float * partial_dW;
  float * partial_dc;
  float * partial_db;
  void init()
  {
    partial_err = 0;
    partial_dW = new float[h*v];
    for(int i=0;i<h*v;i++)partial_dW[i]=0;
    partial_dc = new float[h];
    for(int i=0;i<h;i++)partial_dc[i]=0;
    partial_db = new float[v];
    for(int i=0;i<v;i++)partial_db[i]=0;
  }
  void destroy()
  {
    delete [] partial_dW;
    delete [] partial_dc;
    delete [] partial_db;
  }
  void globalUpdate()
  {
    for(int i=0;i<h*v;i++)
        dW[i] += partial_dW[i];
    for(int i=0;i<h;i++)
        dc[i] += partial_dc[i];
    for(int i=0;i<v;i++)
        db[i] += partial_db[i];
  }
};

void gradient_worker(gradient_info * g,std::vector<long> const & vrtx)
{
  float factor = 1.0f / g->n;
  float factorv= 1.0f / (g->v*g->v);
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t];
    for(long i=0;i<g->v;i++)
    {
      for(long j=0;j<g->h;j++)
      {
        g->partial_dW[i*g->h+j] -= factor * (g->vis0[k*g->v+i]*g->hid0[k*g->h+j] - g->vis[k*g->v+i]*g->hid[k*g->h+j]);
      }
    }

    for(long j=0;j<g->h;j++)
    {
      g->partial_dc[j] -= factor * (g->hid0[k*g->h+j]*g->hid0[k*g->h+j] - g->hid[k*g->h+j]*g->hid[k*g->h+j]);
    }

    for(long i=0;i<g->v;i++)
    {
      g->partial_db[i] -= factor * (g->vis0[k*g->v+i]*g->vis0[k*g->v+i] - g->vis[k*g->v+i]*g->vis[k*g->v+i]);
    }

    for(long i=0;i<g->v;i++)
    {
      g->partial_err += factorv * (g->vis0[k*g->v+i]-g->vis[k*g->v+i])*(g->vis0[k*g->v+i]-g->vis[k*g->v+i]);
    }
  }
}

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

  float * vis0;
  float * hid0;
  float * vis;
  float * hid;
  float * dW;
  float * dc;
  float * db;

  RBM(long _v,long _h,float * _W,float * _b,float * _c,long _n,float * _X)
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

    vis0 = NULL;
    hid0 = NULL;
    vis = NULL;
    hid = NULL;
    dW = NULL;
    dc = NULL;
    db = NULL;
  }
  RBM(long _v,long _h,long _n,float* _X)
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

    vis0 = NULL;
    hid0 = NULL;
    vis = NULL;
    hid = NULL;
    dW = NULL;
    dc = NULL;
    db = NULL;
  }

  void init(int offset)
  {
    boost::posix_time::ptime time_start(boost::posix_time::microsec_clock::local_time());
    if(vis0==NULL)vis0 = new float[n*v];
    if(hid0==NULL)hid0 = new float[n*h];
    if(vis==NULL)vis = new float[n*v];
    if(hid==NULL)hid = new float[n*h];
    if(dW==NULL)dW = new float[h*v];
    if(dc==NULL)dc = new float[h];
    if(db==NULL)db = new float[v];

    std::cout << "n*v=" << n*v << std::endl;
    std::cout << "offset=" << offset << std::endl;
    for(long i=0,size=n*v;i<size;i++)
    {
      vis0[i] = X[i+offset];
    }

    vis2hid(vis0,hid0);
    boost::posix_time::ptime time_end(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration(time_end - time_start);
    std::cout << "init timing:" << duration << '\n';
  }

  void cd(long nGS,float epsilon,int offset=0,bool bottleneck=false)
  {
    boost::posix_time::ptime time_0(boost::posix_time::microsec_clock::local_time());
    std::cout << "cd" << std::endl;

    // CD Contrastive divergence (Hlongon's CD(k))
    //   [dW, db, dc, act] = cd(self, X) returns the gradients of
    //   the weihgts, visible and hidden biases using Hlongon's
    //   approximated CD. The sum of the average hidden units
    //   activity is returned in act as well.

    for(long i=0;i<n*h;i++)
    {
      hid[i] = hid0[i];
    }
    boost::posix_time::ptime time_1(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration10(time_1 - time_0);
    std::cout << "cd timing 1:" << duration10 << '\n';

    for (long iter = 1;iter<=nGS;iter++)
    {
      std::cout << "iter=" << iter << std::endl;
      // sampling
      hid2vis(hid,vis);
      vis2hid(vis,hid);

      long off = dat_offset%(n);
      long offv = off*v;
      long offh = off*h;
      long off_preview = off*(3*WIN*WIN+10);
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
          vis1_preview[k] = orig_arr[offset+off_preview+k];
          vis1_previewG[k] = orig_arr[offset+off_preview+k+WIN*WIN];
          vis1_previewB[k] = orig_arr[offset+off_preview+k+2*WIN*WIN];
        }
      }
      for(long x=0,k=0;x<WIN;x++)
      {
        for(long y=0;y<WIN;y++,k++)
        {
          vis0_preview[k] = vis0[offv+k];
          vis0_previewG[k] = vis0[offv+k+WIN*WIN];
          vis0_previewB[k] = vis0[offv+k+2*WIN*WIN];
        }
      }

    }
    boost::posix_time::ptime time_2(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration21(time_2 - time_1);
    std::cout << "cd timing 2:" << duration21 << '\n';
  
    zero(dW,v*h);
    zero(dc,h);
    zero(db,v);
    boost::posix_time::ptime time_3(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration32(time_3 - time_2);
    std::cout << "cd timing 3:" << duration32 << '\n';
    float * err = new float(0);
    gradient_update(n,vis0,hid0,vis,hid,dW,dc,db,err);
    boost::posix_time::ptime time_4(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration43(time_4 - time_3);
    std::cout << "cd timing 4:" << duration43 << '\n';
    *err = sqrt(*err);
    for(int t=2;t<3&&t<errs.size();t++)
      *err += (errs[errs.size()+1-t]-*err)/t;
    errs.push_back(*err);
    boost::posix_time::ptime time_5(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration54(time_5 - time_4);
    std::cout << "cd timing 5:" << duration54 << '\n';
    std::cout << "epsilon = " << epsilon << std::endl;
    add(W,dW,-epsilon,v*h);
    add(c,dc,-epsilon,h);
    add(b,db,-epsilon,v);

    std::cout << "dW norm = " << norm(dW,v*h) << std::endl;
    std::cout << "dc norm = " << norm(dc,h) << std::endl;
    std::cout << "db norm = " << norm(db,v) << std::endl;
    std::cout << "W norm = " << norm(W,v*h) << std::endl;
    std::cout << "c norm = " << norm(c,h) << std::endl;
    std::cout << "b norm = " << norm(b,v) << std::endl;
    std::cout << "err = " << *err << std::endl;
    delete err;

    boost::posix_time::ptime time_6(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration65(time_6 - time_5);
    std::cout << "cd timing 6:" << duration65 << '\n';
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
    std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
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
    threads.clear();
    vrtx.clear();
  }

  void gradient_update(long n,float * vis0,float * hid0,float * vis,float * hid,float * dW,float * dc,float * db,float * err)
  {
    boost::posix_time::ptime time_0(boost::posix_time::microsec_clock::local_time());

    std::vector<boost::thread*> threads;
    std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
    std::vector<gradient_info*> g;

    boost::posix_time::ptime time_1(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration10(time_1 - time_0);
    std::cout << "gradient update timing 1:" << duration10 << '\n';

    for(long i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    boost::posix_time::ptime time_2(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration21(time_2 - time_1);
    std::cout << "gradient update timing 2:" << duration21 << '\n';
    for(long i=0;i<vrtx.size();i++)
    {
      g.push_back(new gradient_info());
    }
    boost::posix_time::ptime time_3(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration32(time_3 - time_2);
    std::cout << "gradient update timing 3:" << duration32 << '\n';
    for(long thread=0;thread<vrtx.size();thread++)
    {
      g[thread]->n = n;
      g[thread]->v = v;
      g[thread]->h = h;
      g[thread]->vis0 = vis0;
      g[thread]->hid0 = hid0;
      g[thread]->vis = vis;
      g[thread]->hid = hid;
      g[thread]->dW = dW;
      g[thread]->dc = dc;
      g[thread]->db = db;
      g[thread]->init();
      threads.push_back(new boost::thread(gradient_worker,g[thread],vrtx[thread]));
    }
    boost::posix_time::ptime time_4(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration43(time_4 - time_3);
    std::cout << "gradient update timing 4:" << duration43 << '\n';
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
      g[thread]->globalUpdate();
      *err += g[thread]->partial_err;
      g[thread]->destroy();
      delete g[thread];
    }
    boost::posix_time::ptime time_5(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration54(time_5 - time_4);
    std::cout << "gradient update timing 5:" << duration54 << '\n';
    threads.clear();
    vrtx.clear();
    g.clear();
  }
  
  void hid2vis(const float * H,float * V)
  {
    std::vector<boost::thread*> threads;
    std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
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
    threads.clear();
    vrtx.clear();
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
  long num_iters;
  long batch_iter;
  DataUnit(long _v,long _h,long _num_iters = 100,long _batch_iter = 1)
  {
    num_iters = _num_iters;
    batch_iter = _batch_iter;
    v = _v;
    h = _h;
    W = new float[v*h];
    b = new float[v];
    c = new float[h];
    constant(c,0.5f,h);
    constant(b,0.5f,v);
    constant(W,0.5f,v*h);
      hidden = NULL;
     visible = NULL;
    visible0 = NULL;
  }

  void train(float * dat, long n, long total_n,int n_cd,float epsilon)
  {
    // RBM(long _v,long _h,float * _W,float * _b,float * _c,long _n,float * _X)
    rbm = new RBM(v,h,W,b,c,n,dat);
    for(long i=0;i<num_iters;i++)
    {
      std::cout << "DataUnit::train i=" << i << std::endl;
      long offset = (rand()%(total_n-n));
      for(long k=0;k<batch_iter;k++)
      {
        rbm->init(offset);
        std::cout << "prog:" << 100*(float)k/batch_iter << "%" << std::endl;
        rbm->cd(n_cd,epsilon,offset*(10+3*WIN*WIN));
      }
    }
  }

  void transform(float* X,float* Y)
  {
    rbm->vis2hid(X,Y);
  }

  void initialize_weights(DataUnit* d)
  {
    if(v==d->h&&h==d->v)
    {
      for(int i=0;i<v;i++)
      {
        for(int j=0;j<h;j++)
        {
          W[i*h+j] = d->W[j*d->h+i];
        }
      }
    }
  }

  void initialize_weights(DataUnit* d1,DataUnit* d2)
  {
    if(v==d1->h+d2->h&&d1->v==h&&d2->v==h)
    {
      std::cout << "initialize bottleneck" << std::endl;
      char ch;
      std::cin >> ch;
      int j=0;
      for(int k=0;j<d1->h;j++,k++)
      {
        for(int i=0;i<d1->v;i++)
        {
          W[i*h+j] = d1->W[k*d1->h+i];
        }
      }
      for(int k=0;j<d1->h+d2->h;j++,k++)
      {
        for(int i=0;i<d2->v;i++)
        {
          W[i*h+j] = d2->W[k*d2->h+i];
        }
      }
    }
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
    bottle_neck = NULL;
  }
  void copy(float * X,float * Y,long num)
  {
    for(long i=0;i<num;i++)
    {
      Y[i] = X[i];
    }
  }
  void train(long in_num,long out_num,long n_samp,long total_n,long n_cd,float epsilon,float * in,float * out)
  {
    float * X = NULL;
    float * Y = NULL;
    float * IN = NULL;
    float * OUT = NULL;
    X = new float[in_num*n_samp];
    IN = new float[in_num*n_samp];
    for(long i=0;i<in_num*n_samp;i++)
    {
      X[i] = in[i];
    }
    for(long i=0;i<input_branch.size();i++)
    {
      if(i>0)input_branch[i]->initialize_weights(input_branch[i-1]); // initialize weights to transpose of previous layer weights M_i -> W = M_{i-1} -> W ^ T
      input_branch[i]->train(X,n_samp,total_n,n_cd,epsilon);
      Y = new float[input_branch[i]->h*n_samp];
      input_branch[i]->transform(X,Y);
      delete [] X;
      X = NULL;
      std::cout << "X init:" << in_num*n_samp << "    " << "X fin:" << input_branch[i]->h*n_samp << std::endl;
      X = new float[input_branch[i]->h*n_samp];
      copy(Y,X,input_branch[i]->h*n_samp);
      copy(Y,IN,input_branch[i]->h*n_samp);
      delete [] Y;
      Y = NULL;
    }
    delete [] X;
    X = NULL;
    X = new float[out_num*n_samp];
    OUT = new float[in_num*n_samp];
    for(long i=0;i<out_num*n_samp;i++)
    {
      X[i] = out[i];
    }
    for(long i=0;i<output_branch.size();i++)
    {
      if(i>0)output_branch[i]->initialize_weights(output_branch[i-1]); // initialize weights to transpose of previous layer weights M_i -> W = M_{i-1} -> W ^ T
      output_branch[i]->train(X,n_samp,total_n,n_cd,epsilon);
      Y = new float[output_branch[i]->h*n_samp];
      output_branch[i]->transform(X,Y);
      delete [] X;
      X = NULL;
      X = new float[output_branch[i]->h*n_samp];
      copy(Y,X,output_branch[i]->h*n_samp);
      copy(Y,OUT,output_branch[i]->h*n_samp);
      delete [] Y;
      Y = NULL;
    }
    delete [] X;
    X = NULL;
    if(bottle_neck!=NULL)
    {
      X = new float[(input_branch[input_branch.size()-1]->h + output_branch[output_branch.size()-1]->h)*n_samp];
      for(long s=0;s<n_samp;s++)
      {
        long i=0;
        for(long k=0;i<in_num&&k<in_num;i++,k++)
        {
          X[s*(in_num+out_num)+i] = IN[s*in_num+k];
        }
        for(long k=0;i<in_num+out_num&&k<out_num;i++,k++)
        {
          X[s*(in_num+out_num)+i] = OUT[s*out_num+k];
        }
      }
      //bottle_neck->initialize_weights(input_branch[input_branch.size()-1],output_branch[output_branch.size()-1]); // initialize weights to transpose of previous layer weights M_i -> W = M_{i-1} -> W ^ T
      bottle_neck->train(X,n_samp,total_n,n_cd,epsilon);
      delete [] X;
      X = NULL;
    }
    delete [] IN;
    IN = NULL;
    delete [] OUT;
    OUT = NULL;
  }
};

RBM * rbm = NULL;

mRBM * mrbm = NULL;

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
      rbm->init(offset);
      std::cout << "prog:" << 100*(float)k/batch_iter << "%" << std::endl;
      rbm->cd(n_cd,c_epsilon,offset*(10+3*WIN*WIN));
    }
  }
}

struct mrbm_params
{

  long batch_iter;
  long num_batch;
  long total_n;
  long n;
  float epsilon;
  long n_iter;
  long n_cd;

  long v;
  long h;

  std::vector<long> input_sizes;

  std::vector<long> output_sizes;

  std::vector<long> input_iters;

  std::vector<long> output_iters;

  long bottleneck_iters;

  mrbm_params()
  {

    v  = 3*WIN*WIN+10;
    h  = 3*WIN*WIN+10;

    long h1 = 3*WIN*WIN+10;
    long h2 = 3*WIN*WIN+10;
    long h3 = 3*WIN*WIN+10;
    long h4 = 3*WIN*WIN+10;
    long h5 = 3*WIN*WIN+10;
    long h6 = 3*WIN*WIN+10;
    long h7 = 3*WIN*WIN+10;
    long h8 = 3*WIN*WIN+10;
    long h9 = 3*WIN*WIN+10;
    long h10= 3*WIN*WIN+10;

    n_cd = 1;
    num_batch = n_batch;
    batch_iter = 1;
    n = n_samples;
    total_n = n_samples;
    epsilon = c_epsilon;
    n_iter = 100;

    input_sizes.push_back(v);
    input_sizes.push_back(h1);
    input_sizes.push_back(h2);
    input_sizes.push_back(h3);
    input_sizes.push_back(h4);
    input_sizes.push_back(h5);
    input_sizes.push_back(h6);
    input_sizes.push_back(h7);
    input_sizes.push_back(h8);
    input_sizes.push_back(h9);
    input_sizes.push_back(h10);

    output_sizes.push_back(v);
    output_sizes.push_back(h1);
    output_sizes.push_back(h2);
    output_sizes.push_back(h3);
    output_sizes.push_back(h4);
    output_sizes.push_back(h5);
    output_sizes.push_back(h6);
    output_sizes.push_back(h7);
    output_sizes.push_back(h8);
    output_sizes.push_back(h9);
    output_sizes.push_back(h10);

    input_iters.push_back(100);
    input_iters.push_back(20);
    input_iters.push_back(20);
    input_iters.push_back(20);
    input_iters.push_back(20);
    input_iters.push_back(20);
    input_iters.push_back(20);
    input_iters.push_back(20);
    input_iters.push_back(20);
    input_iters.push_back(20);

    output_iters.push_back(100);
    output_iters.push_back(20);
    output_iters.push_back(20);
    output_iters.push_back(20);
    output_iters.push_back(20);
    output_iters.push_back(20);
    output_iters.push_back(20);
    output_iters.push_back(20);
    output_iters.push_back(20);
    output_iters.push_back(20);

    bottleneck_iters = 300;

  }
};

void run_mrbm(mrbm_params p,float * dat_in,float * dat_out)
{
  mrbm = new mRBM();
  for(long i=0;i+1<p.input_sizes.size();i++)
  {
    mrbm->input_branch.push_back(new DataUnit(p.input_sizes[i],p.input_sizes[i+1],p.input_iters[i]));
  }
  for(long i=0;i+1<p.output_sizes.size();i++)
  {
    mrbm->output_branch.push_back(new DataUnit(p.output_sizes[i],p.output_sizes[i+1],p.output_iters[i]));
  }
  long bottle_neck_num = (p.input_sizes[p.input_sizes.size()-1]+p.output_sizes[p.output_sizes.size()-1]);
  mrbm->bottle_neck = new DataUnit(p.input_sizes[p.input_sizes.size()-1]+p.output_sizes[p.output_sizes.size()-1],bottle_neck_num,p.bottleneck_iters);
  mrbm->train(p.v,p.h,p.num_batch,p.total_n,p.n_cd,p.epsilon,dat_in,dat_out);
  std::cout << "training done..." << std::endl;
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

void keyboard(unsigned char Key, int x, int y)
{
  switch(Key)
  {
    case 'w':dat_offset++;break;
    case 's':dat_offset--;if(dat_offset<0)dat_offset=0;break;
    case ' ':
      {

        mrbm_params params;

        boost::thread * thr ( new boost::thread ( run_mrbm
                                                , params
                                                , img_arr
                                                , out_arr
                                                ) 
                            );
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
  //readBinary( "cifar-10-batches-bin/data_batch_1.bin", data );
  //readBinary( "cifar-10-batches-bin/data_batch_2.bin", data );
  //readBinary( "cifar-10-batches-bin/data_batch_3.bin", data );
  //readBinary( "cifar-10-batches-bin/data_batch_4.bin", data );
  //readBinary( "cifar-10-batches-bin/data_batch_5.bin", data );
  //readBinary( "cifar-10-batches-bin/test_batch.bin", data );

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

  n_samples = 4;//data.size() / (3*WIN*WIN+1);
  n_batch = n_samples - 1;

  // visible = 3*WIN*WIN
  // hidden = 3*WIN*WIN + 10
  n_variables = ((3*WIN*WIN + 10));

  img_arr = new float[n_samples * n_variables];
  out_arr = new float[n_samples * n_variables];
  orig_arr = new float[n_samples * n_variables];

  for(long s=0,k=0,t=0;s<n_samples;s++)
  {
    for(long i=0;i<3*WIN*WIN;i++,k++,t++)
    {
      img_arr[k] = (unsigned char)data[1+(3*WIN*WIN+1)*s+i]/256.0;
      out_arr[k] = (unsigned char)data[1+(3*WIN*WIN+1)*s+i]/256.0;
    }
    for(long i=0;i<10;i++,k++)
    {
      img_arr[k] = (long)data[(3*WIN*WIN+1)*s]==k;
      out_arr[k] = (long)data[(3*WIN*WIN+1)*s]==k;
    }
    t++;
  }
  for(int i=0;i<n_samples*n_variables;i++)
    orig_arr[i] = img_arr[i];
  SemblanceLayer semb_layer(WIN,WIN);
  for(long s=0;s<n_samples;s++)
  {
    semb_layer(WIN,WIN,&img_arr[(3*WIN*WIN+10)*s],&img_arr[(3*WIN*WIN+10)*s]);
    semb_layer(WIN,WIN,&out_arr[(3*WIN*WIN+10)*s],&out_arr[(3*WIN*WIN+10)*s]);
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

