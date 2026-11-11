/* Matched 256x128 RGB inputs: two 96x80 rectangles on white, colors reversed.
 * Their decoded pixels reproduce the archived PNG reference pair. No labels
 * or extra pixels tell the language decoder which picture it received. */
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr,"usage: %s EXISTING_DIRECTORY\n",argv[0]); return 2; }
    for (int flip=0; flip<2; flip++) {
        char path[4096];
        int n=snprintf(path,sizeof path,"%s/left-%s.ppm",argv[1],flip ? "blue" : "red");
        if (n<0 || (size_t)n>=sizeof path) return 1;
        FILE *f=fopen(path,"wbx");
        if (!f) { perror(path); return 1; }
        int ok=fprintf(f,"P6\n256 128\n255\n")>0;
        for (int y=0; y<128 && ok; y++) for (int x=0; x<256 && ok; x++) {
            unsigned char rgb[3]={255,255,255};
            int left=x>=16 && x<112, right=x>=144 && x<240;
            if (y>=24 && y<104 && (left || right)) {
                int red=left != flip;
                rgb[0]=red ? 253 : 0; rgb[1]=0; rgb[2]=red ? 0 : 254;
            }
            ok=fwrite(rgb,1,3,f)==3;
        }
        if (fclose(f)) ok=0;
        if (!ok) return 1;
    }
    return 0;
}
