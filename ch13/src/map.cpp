#include "myslam/map.h"
#include "myslam/feature.h"

namespace myslam {
    void Map::InsertKeyFrame(Frame::Ptr frame) { // 将一个新的关键帧插入到地图中
        current_frame_ = frame; // 将当前处理的帧设置为传入的帧
        if ( keyframes_.find(frame->keyframe_id_) == keyframes_.end() ) { // std::unordered_map 提供了 .find() 成员函数，用于在关联容器中查找具有指定键的元素，若找不到则指向容器末尾
        // frame->keyframe_id_ 是要查找的键。keyframes_.find(frame->keyframe_id_) 返回一个迭代器，指向具有指定键 frame->keyframe_id_ 的元素，若不存在则指向 keyframes_ 的末尾。
        // 检查输入的关键帧是否已经存在于keyframes_集合中。如果不存在，则执行插入操作。
            // 将关键帧分别插入到keyframes_和active_keyframes_集合中
            keyframes_.insert( make_pair(frame->keyframe_id_, frame) );
            active_keyframes_.insert( make_pair(frame->keyframe_id_, frame) );
            // std::unordered_map 等关联容器提供了 .insert 成员函数，用于向容器中插入新元素。
            // std::make_pair 是一个模板函数，用于创建 std::pair 对象。它接受两个参数，分别用于初始化 std::pair 的 first 和 second 成员，此处分别为ID和特征点对象
        } else {
            // 如果关键帧已存在，则更新其引用。
            keyframes_[frame->keyframe_id_] = frame; // 利用关联容器（如 std::unordered_map 或 std::map）的下标运算符 [] 来访问或插入元素
            active_keyframes_[frame->keyframe_id_] = frame;
        }

        if (active_keyframes_.size() > num_active_keyframes_ ) { // 如果激活关键帧的数量超过了预设的最大值，则调用RemoveOldKeyframe函数来移除旧关键帧
            RemoveOldKeyframe();
        }
    }

    void Map::InsertMapPoint( MapPoint::Ptr map_point ) { // 用于将一个新的地图点插入到地图
        if ( landmarks_.find(map_point->id_) == landmarks_.end() ) { // 检查这个地图点是否已经存在于landmarks_集合
            landmarks_.insert( make_pair(map_point->id_, map_point) );
            active_landmarks_.insert( make_pair(map_point->id_, map_point) );
        }  else {
            landmarks_[map_point->id_] = map_point;
            active_landmarks_[map_point->id_] = map_point;
        }
    }

    void Map::RemoveOldKeyframe() { // 用于移除一个旧的关键帧
        if (current_frame_ == nullptr )  return; // 返回void，即无返回，相当于中断。如果没有当前帧，则直接返回。
        
        // 寻找与当前帧最近和最远的两个关键帧
        double max_dis = 0, min_dis = 9999; // 初始化变当前帧最远和最近的关键帧的距离，以及相应ID
        double max_kf_id = 0, min_kf_id = 9;
        auto Twc = current_frame_->Pose().inverse(); // 计算当前帧的位姿的逆
        for (auto& kf : active_keyframes_ ) { // 遍历所有活跃的关键帧
            if (kf.second == current_frame_) continue; // 跳过当前帧自身。.second不是第二个活跃的关键帧，而是帧结构体，.first是其ID。
            auto dis = (kf.second->Pose() * Twc).log().norm(); // 计算当前帧与每个关键帧之间的相对位姿距离。
            // kf.second->Pose() * Twc 相当于 Trc (当前帧->其他活跃帧)。.log 将SE(3)->se(3)，.norm()计算向量的范数。这里的范数包含了旋转向量的部分。
            if (dis > max_dis) {
                max_dis = dis;
                max_kf_id = kf.first; // 修改距离最大的帧的ID
            }
            if (dis < min_dis) {
                min_dis = dis;
                min_kf_id = kf.first;
            }
        }

        const double min_dis_th = 0.2;  // 最近阈值。用于判断关键帧是否足够接近当前帧。
        Frame::Ptr frame_to_remove = nullptr; // 初始化要移除的关键帧的指针
        if (min_dis < min_dis_th ) {
            // 如果存在很近的帧，优先删除最近的
            frame_to_remove = keyframes_.at(min_kf_id); // .at()为 std::unordered_map的成员函数，用于通过键访问元素。如果键不存在，它会抛出一个异常。此处用于安全地访问已知存在的关键帧。
        }  else {
            // 删除最远的
            frame_to_remove = keyframes_.at(max_kf_id);
        }

        // remove keyframe and landmark observation
        LOG(INFO) << "remove keyframe " << frame_to_remove->keyframe_id_; // LOG(INFO) 是一个日志宏或函数，用于输出信息级别的日志。具体实现可能依赖于使用的日志库。
        active_keyframes_.erase( frame_to_remove->keyframe_id_ ); // 从活跃关键帧集合中移除该关键帧。
        for ( auto feat : frame_to_remove->features_left_ ) { // 遍历要移除的关键帧中的左侧特征点。
            // 对于每个特征点，如果它关联了一个地图点，则从该地图点中移除该特征点的观测。
            auto mp = feat->map_point_.lock(); // 声明一个自动变量mp，并将其初始化为feat对象的map_point_成员(weak_ptr)调用lock()方法的结果
            // 若存在原始shared_ptr，返回一个指向相同对象的新的shared_ptr实例（不影响原始shared_ptr的生命周期） 
            // 若不存在原始shared_ptr，返回一个空的shared_ptr ，在if()中会判断为假。
            if (mp) {
                mp->RemoveObservation(feat);
            }
        }
        for ( auto feat : frame_to_remove->features_right_ ) { // 遍历要移除的关键帧中的右侧特征点。
            if (feat == nullptr )   continue; // 跳过空指针
            // 移除右侧特征点的观测
            auto mp = feat->map_point_.lock();
            if (mp) {
                mp->RemoveObservation(feat);
            }
        }

        CleanMap(); // 调用CleanMap函数清理地图。
    }

    void Map::CleanMap() { // 用于清理地图
        int cnt_landmark_removed = 0; // 初始化被移除的地图点计数
        for ( auto iter = active_landmarks_.begin();  iter != active_landmarks_.end();   ) { // 遍历活跃地图点集合
            if (iter->second->observed_times_ == 0) { // 如果一个地图点没有被任何关键帧观测到，则移除它。
                iter = active_landmarks_.erase(iter);
                cnt_landmark_removed++;
            } else {
                ++iter;
            }
        }

        LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
    }

} // namespace myslam