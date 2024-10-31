#include "myslam/mappoint.h"
#include "myslam/feature.h"

namespace myslam {
    MapPoint::MapPoint( long id, Vec3 position ) : id_(id), pos_(position) {} // 重载的构造函数

    MapPoint::Ptr MapPoint::CreateNewMappoint() { // 工厂模式的实例创建函数。创建实例并返回指针、赋予ID、增加计数。
        static long factory_id = 0;
        MapPoint::Ptr new_mappoint(new MapPoint) ;
        new_mappoint->id_ = factory_id++;
        return new_mappoint;
    }

    // 用于移除一个与地图点相关联的特征点观测
    void MapPoint::RemoveObservation(std::shared_ptr<Feature> feat) {
        std::unique_lock<std::mutex> lck(data_mutex_); // 锁定数据，确保在移除观测时，线程安全。
        for (auto iter = observations_.begin(); iter != observations_.end(); iter++ ) { // 遍历观测列表
            if (iter->lock() == feat ) { // 尝试锁定iter指向的弱指针，并检查它是否等于要移除的特征点feat。
                observations_.erase(iter); // 从observations_中移除指定观测。
                feat->map_point_.reset(); // 将feat中的map_point_成员重置，表示这个特征点不再与任何地图点相关联。
                observed_times_--; // 减少observed_times_成员变量的值，即该地图点被观察到的次数。
                break;
            }
        }
    }
} // namespace myslam
