#include "myslam/frame.h"

namespace myslam {
    Frame::Frame( long id, double time_stamp, const SE3 &pose, const Mat &left, const Mat &right )
        : id_(id), time_stamp_(time_stamp), pose_(pose), left_img_(left), right_img_(right)  {}

    Frame::Ptr Frame::CreateFrame() { // Frame::Ptr 为 Frame 类内部通过typedef定义的智能指针类型别名，指向 std::shared_ptr<Frame>
        static long factory_id = 0;  
        Frame::Ptr new_frame(new Frame); // 新建了一个Frame类型的对象。Frame::Ptr表示智能指针。
        new_frame->id_ = factory_id++; // 静态成员函数只能访问静态变量，或可通过内部定义的类的实例访问该实例的非静态成员变量。
        return new_frame;
    }

    void Frame::SetKeyFrame() {
        static long keyframe_factory_id = 0; // 成员函数内部的static变量，只能在调用该函数时访问，不过不同实例调用时，访问的static变量是同一个变量实例。
        is_keyframe_ = true;
        keyframe_id_ = keyframe_factory_id++;
    }
}