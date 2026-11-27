// CPSP-style exact search, 3D cubic HP. usage: cpsp SEQ X Y Z Emin
// 1) enumerate H-core cell sets (|core|=#H) in X*Y*Z box with >=Emin edges (touching min faces => translation-unique)
// 2) thread chain: H residues exactly on core cells, P residues off core. Report max contacts feasible.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define M 40
static int X,Y,Z,C,N,Emin,n; static char seq[128]; static int H[128];
static int nb[256][6],nn[256]; static unsigned char in[256];
static long long ncores=0,nthread=0; static int bestE=-1, foundE[64];
static int cx[256],cy[256],cz[256];
static unsigned char occ[M][M][M], core[M][M][M]; static int px[128],py[128],pz[128];
static int D[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
static int hh; static int sol[128][3]; static int corecells[256],ccount; static long long tn;
static int thread_rec(int j,int hused){
  tn++;
  if(j==n) return 1;
  for(int d=0;d<6;d++){
    int x=px[j-1]+D[d][0],y=py[j-1]+D[d][1],z=pz[j-1]+D[d][2];
    if(occ[x][y][z]) continue;
    if(H[j]!=core[x][y][z]) continue; // H<->core
    px[j]=x;py[j]=y;pz[j]=z;occ[x][y][z]=1;
    if(thread_rec(j+1,hused+H[j])){ for(int i=0;i<n;i++){sol[i][0]=px[i];sol[i][1]=py[i];sol[i][2]=pz[i];} occ[x][y][z]=0; return 1;}
    occ[x][y][z]=0;
  }
  return 0;
}
static int try_core(int e){
  // place core in grid with offset 15; residue 0 on each core cell (if H[0]) else anywhere near -> assume seq[0]=='H' or handle by start cell loop over neighbors
  int o=15; memset(core,0,sizeof core);
  for(int c=0;c<C;c++) if(in[c]) core[o+cx[c]][o+cy[c]][o+cz[c]]=1;
  // first residue: if H, on core cell; else on any cell adjacent to core region (try all cells in box+1 margin, off core)
  for(int x=o-1;x<=o+X;x++)for(int y=o-1;y<=o+Y;y++)for(int z=o-1;z<=o+Z;z++){
    if(H[0]!=core[x][y][z]) continue;
    px[0]=x;py[0]=y;pz[0]=z;occ[x][y][z]=1;
    int r=thread_rec(1,H[0]); occ[x][y][z]=0;
    if(r) return 1;
  }
  return 0;
}
static void dfs(int start,int k,int e,int minx,int miny,int minz){
  if(k==N){
    if(e<Emin) return; if(minx|miny|minz) return; // require touching x=0,y=0,z=0 planes handled via flags below
    return;
  }
}
// simple recursion with touch flags
static int touch[3];
static void rec(int start,int k,int e){
  if(k==N){
    if(e<Emin) return;
    // touches min faces?
    int tx=0,ty=0,tz=0; for(int c=0;c<C;c++) if(in[c]){ if(cx[c]==0)tx=1; if(cy[c]==0)ty=1; if(cz[c]==0)tz=1; }
    if(!(tx&&ty&&tz)) return;
    ncores++;
    if(try_core(e)){ nthread++; foundE[e]++; if(e>bestE){bestE=e; } if(foundE[e]==1){ printf("THREADS E=%d contacts=%d\n",e,e-hh); for(int i=0;i<n;i++)printf("(%d,%d,%d) ",sol[i][0]-15,sol[i][1]-15,sol[i][2]-15); printf("\n"); fflush(stdout);} }
    return;
  }
  if(C-start<N-k) return;
  if(e+3*(N-k)<Emin) return;
  for(int c=start;c<C;c++){
    int a=0; for(int q=0;q<nn[c];q++) if(in[nb[c][q]]) a++;
    in[c]=1; rec(c+1,k+1,e+a); in[c]=0;
  }
}
int main(int argc,char**argv){
  strcpy(seq,argv[1]); X=atoi(argv[2]);Y=atoi(argv[3]);Z=atoi(argv[4]);Emin=atoi(argv[5]);
  n=strlen(seq); N=0; for(int i=0;i<n;i++){H[i]=seq[i]=='H'; N+=H[i];}
  hh=0; for(int i=0;i+1<n;i++) if(H[i]&&H[i+1]) hh++;
  C=X*Y*Z;
  for(int x=0;x<X;x++)for(int y=0;y<Y;y++)for(int z=0;z<Z;z++){int c=(x*Y+y)*Z+z;cx[c]=x;cy[c]=y;cz[c]=z;nn[c]=0;
    for(int i=0;i<6;i++){int a=x+D[i][0],b=y+D[i][1],cc=z+D[i][2]; if(a>=0&&a<X&&b>=0&&b<Y&&cc>=0&&cc<Z) nb[c][nn[c]++]=(a*Y+b)*Z+cc;}}
  rec(0,0,0);
  printf("seq=%s #H=%d HHbonds=%d box=%dx%dx%d Emin=%d cores=%lld threadable=%lld bestE=%d => best contacts=%d\n",seq,N,hh,X,Y,Z,Emin,ncores,nthread,bestE,bestE<0?-1:bestE-hh);
  for(int e=0;e<64;e++) if(foundE[e]) printf(" E=%d: %d cores thread\n",e,foundE[e]);
}
