// SPDX-License-Identifier: Apache-2.0
// Diagnostic CUPTI kernel-time snapshots. Not benchmark timings.
#include <cupti_activity.h>
#include <cupti_result.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include <cxxabi.h>

namespace {
struct Entry { unsigned long long count=0, ns=0; };
struct State {
    std::mutex mutex;
    std::map<std::string, Entry> kernels;
    size_t dropped=0;
    std::string output;
};
State &state() { static State *s = new State; return *s; }
void check_result(CUptiResult result, const char *operation) {
    if (result == CUPTI_SUCCESS) return;
    const char *message=nullptr;
    cuptiGetResultString(result,&message);
    std::fprintf(stderr,"QWEN_PROFILE_ERROR: %s: %s\n",operation,message ? message : "unknown CUPTI error");
    std::_Exit(86);
}
#define check(call) check_result((call), #call)
void CUPTIAPI requested(uint8_t **buffer,size_t *size,size_t *max_records) {
    *size=4*1024*1024; *max_records=0;
    void *allocation=nullptr;
    if (posix_memalign(&allocation,8,*size)!=0) std::_Exit(86);
    *buffer=static_cast<uint8_t *>(allocation);
}
void CUPTIAPI completed(CUcontext context,uint32_t stream,uint8_t *buffer,size_t,size_t valid) {
    CUpti_Activity *record=nullptr;
    while (true) {
        const auto result=cuptiActivityGetNextRecord(buffer,valid,&record);
        if (result==CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        check(result);
        if (record->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || record->kind==CUPTI_ACTIVITY_KIND_KERNEL) {
            const auto *kernel=reinterpret_cast<const CUpti_ActivityKernel10 *>(record);
            if (kernel->start && kernel->end>=kernel->start) {
                std::lock_guard<std::mutex> guard(state().mutex);
                auto &entry=state().kernels[kernel->name];
                ++entry.count; entry.ns+=kernel->end-kernel->start;
            }
        }
    }
    size_t dropped=0; check(cuptiActivityGetNumDroppedRecords(context,stream,&dropped));
    { std::lock_guard<std::mutex> guard(state().mutex); state().dropped+=dropped; }
    std::free(buffer);
}
void snapshot() {
    std::lock_guard<std::mutex> guard(state().mutex);
    std::vector<std::pair<std::string,Entry>> rows(state().kernels.begin(),state().kernels.end());
    std::sort(rows.begin(),rows.end(),[](const auto &a,const auto &b){return a.second.ns>b.second.ns;});
    // Atomic publication: a killed server leaves its last complete snapshot.
    const std::string tmp=state().output+".tmp";
    std::ofstream out(tmp);
    out << "# Partial cumulative kernel profile including loading/warmup; not request latency\n# dropped_records=" << state().dropped << "\nname,calls,total_ms\n";
    for (const auto &[name,entry]:rows) {
        int status=0; char *demangled=abi::__cxa_demangle(name.c_str(),nullptr,nullptr,&status);
        std::string readable=status==0 ? demangled : name;
        std::free(demangled);
        std::string escaped;
        for (char c:readable) { if(c=='"') escaped+='"'; escaped+=c; }
        out << '"' << escaped << "\"," << entry.count << ',' << entry.ns/1000000.0 << '\n';
    }
    out.close();
    if (!out || std::rename(tmp.c_str(),state().output.c_str())!=0) std::_Exit(86);
}
__attribute__((constructor)) void initialize() {
    const char *target=std::getenv("QWEN_PROFILE_TARGET");
    const char *output=std::getenv("QWEN_PROFILE_OUTPUT");
    if (!target || !output) return;
    char executable[4096];
    const ssize_t n=readlink("/proc/self/exe",executable,sizeof(executable)-1);
    if (n<0) return;
    executable[n]=0;
    const char *name=std::strrchr(executable,'/');
    if (std::strcmp(name ? name+1 : executable,target)!=0) return;
    state().output=output;
    check(cuptiActivityRegisterCallbacks(requested,completed));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    std::thread([] {
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            check(cuptiActivityFlushAll(0));
            snapshot();
        }
    }).detach();
}
}
