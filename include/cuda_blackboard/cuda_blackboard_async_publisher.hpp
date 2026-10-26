
#pragma once

#include "cuda_blackboard/cuda_blackboard_publisher.hpp"
#include "cuda_blackboard/publish_mailbox.hpp"

#include <rclcpp/rclcpp.hpp>

#include <pthread.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace cuda_blackboard
{

/// Creates a thread from a given thread body. Lets callers hand worker threads to an external
/// thread manager, for example to control their scheduling policy.
using thread_factory_t = std::function<std::thread(std::function<void()> && thread_body)>;

/// The default thread factory, which creates plain `std::thread`s.
struct StdThreadFactory
{
  std::thread operator()(std::function<void()> && thread_body) const
  {
    return std::thread(std::move(thread_body));
  }
};

/// A CudaBlackboardPublisher whose publish() returns at once: the message goes into a single-slot
/// mailbox and a worker thread owned by this object does the publish (D2H copy and serialization
/// for ROS subscribers). A message the worker has not taken yet is replaced by the newer one and
/// counted in dropped(). The destructor stops and joins the worker, so the thread never outlives
/// the publisher. Subscriber counts are the wrapped publisher's.
template <typename T>
class CudaBlackboardAsyncPublisher
{
public:
  /// @param thread_factory Creates the worker thread. Must not be null and must return a joinable
  /// thread. A node under the Agnocast CIE thread configurator must pass a factory that calls
  /// `agnocast_cie_thread_configurator::spawn_non_ros2_thread`. A plain `std::thread` is invisible
  /// to the configurator and keeps the scheduling of the thread that constructs this publisher.
  CudaBlackboardAsyncPublisher(
    rclcpp::Node & node, const std::string & topic_name,
    thread_factory_t thread_factory = StdThreadFactory{})
  : logger_(node.get_logger()), publisher_(node, topic_name)
  {
    if (!thread_factory) {
      throw std::invalid_argument("Thread factory must not be null");
    }
    // Last, so that a throwing constructor never leaves a joinable thread behind.
    worker_ = thread_factory([this, topic_name] { run(topic_name); });
    if (!worker_.joinable()) {
      throw std::invalid_argument("Thread factory must return a joinable thread");
    }
  }

  ~CudaBlackboardAsyncPublisher()
  {
    mailbox_.stop();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  CudaBlackboardAsyncPublisher(const CudaBlackboardAsyncPublisher &) = delete;
  CudaBlackboardAsyncPublisher & operator=(const CudaBlackboardAsyncPublisher &) = delete;

  /// Hands the message to the worker and returns; a message still waiting there is replaced.
  void publish(std::unique_ptr<const T> msg) { mailbox_.deposit(std::move(msg)); }

  std::size_t get_subscription_count() const { return publisher_.get_subscription_count(); }
  std::size_t get_intra_process_subscription_count() const
  {
    return publisher_.get_intra_process_subscription_count();
  }

  /// Messages replaced before the worker published them (cumulative).
  std::uint64_t dropped() const { return mailbox_.dropped(); }

private:
  void run(const std::string & topic_name)
  {
    // Linux limits thread names to 15 characters: keep the last topic segment.
    const std::string name = topic_name.substr(topic_name.rfind('/') + 1).substr(0, 15);
    pthread_setname_np(pthread_self(), name.c_str());
    while (auto msg = mailbox_.take()) {
      try {
        publisher_.publish(std::move(*msg));
      } catch (const std::exception & e) {
        RCLCPP_ERROR(logger_, "Publishing %s failed: %s", topic_name.c_str(), e.what());
      }
    }
  }

  rclcpp::Logger logger_;
  CudaBlackboardPublisher<T> publisher_;
  PublishMailbox<std::unique_ptr<const T>> mailbox_;
  std::thread worker_;  // declared last: joined in the destructor before the members above go
};

}  // namespace cuda_blackboard
