#pragma once

#ifndef MYSLAM_CONFIG_H
#define MYSLAM_CONFIG_H

#include "myslam/common_include.h"

namespace myslam {
    /** 
     * 配置类，使用 SetParameterFile 确定配置文件，再使用 Get 得到对应值
     * 单例（Singleton）模式：确保一个类只有一个实例，并提供一个全局访问点来获取该实例。通过私有构造模式设置，防止外部代码通过 new 来创建该类的多个实例（构造函数是私有的，外部代码无法调用、实例化 Config 类）
     * * 单例模式的实例，通常用 static 函数检查并创建（而非构造函数）。此处 Config 类通过 static bool SetParameterFile() 构造实例。
     */
    class Config {
        private:
        static std::shared_ptr<Config> config_;
        cv::FileStorage file_;

        Config() {} // private constructor makes a singleton. 
        // 私有构造函数，用于设定单例模式

        public:
        ~Config(); // 析构函数

        static bool SetParameterFile(const std::string &filename); // set a new config file
        // 具体定义：如果 config_ 是空的（即还没有创建实例），则创建一个新的 Config 实例。此处实例存放在 config_ 变量中。

        template <typename T>
        static T Get( const std::string &key ) { // access the parameter values
            return T(Config::config_->file_[key]); // 这里字符串类型的 key 变量，是配置文件.yaml里的变量名
        }

    }; // class Config
}

#endif  // MYSLAM_CONFIG_H