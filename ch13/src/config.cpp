#include "myslam/config.h"

namespace myslam {
    bool Config::SetParameterFile( const std::string &filename ) {
        if (config_ == nullptr ) // 如果 config_ 是空的（即还没有创建实例），则创建一个新的 Config 实例。此处实例存放在 config_ 变量中。
            config_ = std::shared_ptr<Config>(new Config);
        config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ ); // 创建一个 cv::FileStorage 对象，用于读取指定路径的配置文件，并赋予 config_ 实例的成员变量 file_
        // filename.c_str() 将 std::string 转换为 C 风格的字符串（因为 cv::FileStorage 的构造函数接受一个 const char* 类型的参数）
        // cv::FileStorage::READ 指定文件以读取模式打开。

        if (config_->file_.isOpened() == false ) { // 检查 cv::FileStorage 对象是否成功打开了文件。如果 isOpened() 方法返回 false，说明文件打开失败
            LOG(ERROR) << "parameter file " << filename << " does not exist."; // 日志消息记录一个错误消息，指出配置文件不存在。
            config_->file_.release(); // .release() 方法关闭 cv::FileStorage 对象并释放与之关联的资源。
            // 这是必要的，因为即使文件打开失败，file_ 成员变量仍然会持有一个 cv::FileStorage 对象，而这个对象可能会占用系统资源。
            return false;
        }
        return true;
    }

    Config::~Config() {
        if (file_.isOpened()) // 析构函数检查 file_ 成员变量是否仍然持有一个打开的 cv::FileStorage 对象。
            file_.release(); // 如果是，它调用 release() 方法来关闭对象并释放资源。
    }

    std::shared_ptr<Config> Config::config_ = nullptr; // 定义了静态成员变量 config_ 并将其初始化为 nullptr。单例模式中，这个变量用于存储 Config 类的唯一实例

} // namespace myslam