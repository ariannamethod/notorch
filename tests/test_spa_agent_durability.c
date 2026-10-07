/* Compile spa_agent.c separately with open/fstat/mkstemp/fsync/rename/close
 * redirected to the spa_durability_* wrappers below. Other objects retain
 * ordinary POSIX calls. Disable fortified open aliases for this injected object
 * only (-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0), so wrappers observe every call.
 * This exercises real files plus ordered fault injection.
 */
#define _POSIX_C_SOURCE 200809L
#include "spa_agent.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum fault {
    NO_FAULT, OPEN_FAULT, STAT_FAULT, NOT_DIRECTORY, CREATE_FAULT,
    FILE_SYNC_FAULT, RENAME_FAULT, DIRECTORY_SYNC_FAULT, DIRECTORY_CLOSE_FAULT
};
static int tracking, fault, directory_fd=-1, file_fd=-1;
static int wrapper_error, file_synced, installed, directory_synced;
static unsigned checks;
static char events[64];
static size_t event_count;
static const char *expected_parent, *gate;

#define CHECK(c,msg) do { ++checks; if(!(c)) { \
    fprintf(stderr,"FAIL %s:%d: %s (events=%s)\n",gate,__LINE__,msg,events); return 0; } } while(0)

static void event(char value) {
    if(event_count+1>=sizeof(events)) { wrapper_error=1; return; }
    events[event_count++]=value; events[event_count]='\0';
}

int spa_durability_open(const char *path,int flags,...) {
    mode_t mode=0;
    if(flags&O_CREAT) { va_list ap; va_start(ap,flags); mode=(mode_t)va_arg(ap,int); va_end(ap); }
    if(tracking) {
        event('O');
        if(strcmp(path,expected_parent) || (flags&O_CREAT)) wrapper_error=1;
        if(fault==OPEN_FAULT) { errno=EACCES; return -1; }
    }
    int fd=flags&O_CREAT ? open(path,flags,mode) : open(path,flags);
    if(tracking && fd>=0) directory_fd=fd;
    return fd;
}
int spa_durability_fstat(int fd,struct stat *out) {
    if(tracking) {
        event('D');
        if(fd!=directory_fd || file_fd>=0) wrapper_error=1;
        if(fault==STAT_FAULT) { errno=EIO; return -1; }
    }
    int result=fstat(fd,out);
    if(tracking && result==0 && fault==NOT_DIRECTORY) out->st_mode=S_IFREG;
    return result;
}
int spa_durability_mkstemp(char *pattern) {
    if(tracking) {
        event('T');
        if(directory_fd<0 || strcmp(events,"ODT")) wrapper_error=1;
        if(fault==CREATE_FAULT) { errno=ENOSPC; return -1; }
    }
    int fd=mkstemp(pattern);
    if(tracking && fd>=0) file_fd=fd;
    return fd;
}
int spa_durability_fsync(int fd) {
    if(tracking) {
        if(fd==directory_fd) {
            event('P');
            if(!installed || !file_synced) wrapper_error=1;
            if(fault==DIRECTORY_SYNC_FAULT) { errno=EIO; return -1; }
        } else if(fd==file_fd) {
            event('F');
            if(installed) wrapper_error=1;
            if(fault==FILE_SYNC_FAULT) { errno=EIO; return -1; }
        } else wrapper_error=1;
    }
    int result=fsync(fd);
    if(tracking && result==0) {
        if(fd==directory_fd) directory_synced=1; else file_synced=1;
    }
    return result;
}
int spa_durability_rename(const char *from,const char *to) {
    if(tracking) {
        event('R');
        if(!file_synced || fcntl(file_fd,F_GETFD)!=-1 || errno!=EBADF) wrapper_error=1;
        if(fault==RENAME_FAULT) { errno=EACCES; return -1; }
    }
    int result=rename(from,to);
    if(tracking && result==0) installed=1;
    return result;
}
int spa_durability_close(int fd) {
    int is_directory=tracking && fd==directory_fd;
    if(is_directory) event('c');
    int result=close(fd);
    if(is_directory && fault==DIRECTORY_CLOSE_FAULT) { errno=EIO; return -1; }
    return result;
}

static size_t read_bytes(const char *path,unsigned char bytes[4096]) {
    FILE *file=fopen(path,"rb");
    if(!file) return 0;
    size_t n=fread(bytes,1,4096,file);
    int extra=fgetc(file),bad=ferror(file);
    if(fclose(file)!=0) bad=1;
    return bad || extra!=EOF ? 0 : n;
}
static int no_temporary_files(const char *directory) {
    DIR *dir=opendir(directory);
    if(!dir) return 0;
    int clean=1;
    struct dirent *entry;
    while((entry=readdir(dir))) if(strstr(entry->d_name,".tmp.")) clean=0;
    if(closedir(dir)!=0) clean=0;
    return clean;
}
static int run_case(const char *directory,const char *path,const nt_spa_agent *old,
    const nt_spa_agent *replacement,int injected,const char *order,int expected_errno) {
    unsigned char before[4096],after[4096];
    nt_spa_agent loaded,unchanged=*replacement;
    tracking=0;
    CHECK(nt_spa_agent_save(old,path)==NT_SPA_OK,"install original checkpoint");
    size_t before_size=read_bytes(path,before);
    CHECK(before_size==2296,"canonical original remains 2296 bytes");
    directory_fd=-1; file_fd=-1; wrapper_error=0; file_synced=0;
    installed=0; directory_synced=0; event_count=0; events[0]='\0';
    expected_parent=directory; fault=injected; tracking=1; errno=0;
    int result=nt_spa_agent_save(replacement,path),observed_errno=errno;
    tracking=0;
    CHECK(result==(injected==NO_FAULT ? NT_SPA_OK : NT_SPA_E_IO),"reported save status matches injected syscall result");
    CHECK(!wrapper_error && strcmp(events,order)==0,"open/validate/create/file-sync/rename/directory-sync/close order");
    CHECK(!expected_errno || observed_errno==expected_errno,"first syscall errno survives cleanup");
    CHECK(!memcmp(replacement,&unchanged,sizeof(unchanged)),"save leaves resident agent unchanged");
    if(directory_fd>=0) CHECK(fcntl(directory_fd,F_GETFD)==-1 && errno==EBADF,"directory descriptor released");
    if(file_fd>=0) CHECK(fcntl(file_fd,F_GETFD)==-1 && errno==EBADF,"temporary descriptor released");
    CHECK(no_temporary_files(directory),"no temporary files remain");
    CHECK(nt_spa_agent_load(&loaded,path)==NT_SPA_OK,"resident checkpoint remains readable");
    if(injected==NO_FAULT || injected==DIRECTORY_SYNC_FAULT || injected==DIRECTORY_CLOSE_FAULT) {
        CHECK(installed && nt_spa_agent_hash(&loaded)==nt_spa_agent_hash(replacement),"post-rename result retains installed replacement");
        CHECK(directory_synced==(injected!=DIRECTORY_SYNC_FAULT),"directory sync success is recorded precisely");
    } else {
        size_t after_size=read_bytes(path,after);
        CHECK(!installed && after_size==before_size && !memcmp(before,after,before_size),"pre-rename failure preserves exact prior checkpoint bytes");
    }
    return 1;
}

static int relative_path_case(const char *directory,const nt_spa_agent *old,
    const nt_spa_agent *replacement) {
    int cwd=open(".",O_RDONLY);
    CHECK(cwd>=0,"capture current directory");
    if(chdir(directory)!=0) { close(cwd); CHECK(0,"enter private fixture directory"); }
    int result=run_case(".","life.bin",old,replacement,NO_FAULT,"ODTFRPc",0);
    int restored=fchdir(cwd),closed=close(cwd);
    CHECK(restored==0 && closed==0,"restore current directory");
    return result;
}

int main(void) {
    char directory[]="/tmp/notorch-spa-durability-XXXXXX",path[512];
    nt_spa_agent old,replacement;
    nt_spa_agent_config config;
    static const struct {
        const char *name,*order;
        int fault,error;
    } cases[]={
        {"durable successful replacement","ODTFRPc",NO_FAULT,0},
        {"directory open failure before mutation","O",OPEN_FAULT,EACCES},
        {"directory stat failure before mutation","ODc",STAT_FAULT,EIO},
        {"non-directory refused before mutation","ODc",NOT_DIRECTORY,ENOTDIR},
        {"temporary creation failure","ODTc",CREATE_FAULT,ENOSPC},
        {"file sync failure preserves prior file","ODTFc",FILE_SYNC_FAULT,EIO},
        {"rename failure preserves prior file","ODTFRc",RENAME_FAULT,EACCES},
        {"directory sync failure reports installed replacement","ODTFRPc",DIRECTORY_SYNC_FAULT,EIO},
        {"directory close failure reports installed replacement","ODTFRPc",DIRECTORY_CLOSE_FAULT,EIO}
    };
    if(!mkdtemp(directory)) { perror("mkdtemp"); return 1; }
    if(snprintf(path,sizeof(path),"%s/life.bin",directory)<0) return 1;
    nt_spa_agent_config_default(&config);
    if(nt_spa_agent_init(&old,&config)!=NT_SPA_OK) return 1;
    config.seed=2;
    if(nt_spa_agent_init(&replacement,&config)!=NT_SPA_OK) return 1;
    unsigned passed=0;
    for(size_t i=0;i<sizeof(cases)/sizeof(cases[0]);++i) {
        gate=cases[i].name;
        if(run_case(directory,path,&old,&replacement,cases[i].fault,cases[i].order,cases[i].error)) {
            ++passed; printf("PASS %s\n",gate);
        }
    }
    gate="relative path uses current parent directory";
    if(relative_path_case(directory,&old,&replacement)) { ++passed; printf("PASS %s\n",gate); }
    (void)unlink(path); (void)rmdir(directory);
    size_t total=sizeof(cases)/sizeof(cases[0])+1;
    printf("SPA_AGENT_DURABILITY %u/%zu cases, %u checks\n",passed,total,checks);
    return passed==total ? 0 : 1;
}
