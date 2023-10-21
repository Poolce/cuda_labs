#include "taxpy_interface.h"
#include "get_time_result_interface.h"


int main(){
    auto res1 = GetFResultEXP(20, 4, 3, 1.0);
    res1.out();
}