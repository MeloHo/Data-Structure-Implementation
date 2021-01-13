// A simple thread pool implementation
#include <condition_variable>
#include <future>
#include <thread>
#include <vector>
#include <iostream>
#include <queue>
#include <functional>

class ThreadPool
{
public:
	using Task = std::function<void()>;
	
	explicit ThreadPool(std::size_t numThreads)
	{
		start(numThreads);
	}

	~ThreadPool()
	{
		end();
	}
	
	template<typename T>
	auto enqueue(T task) -> std::future<decltype(task())>
	{
		auto wrapper = std::make_shared<std::packaged_taks<decltype(task())()>>(std::move(task));

		{
			std::unique_lock<std::mutex> lock{mEventMutex};
			mTasks.emplace([=]{
				(*wrapper)();
			})
		}
		
		mEventVar.notify_one();
		return wrapper->get_future();
	}

private:
	std::vector<std::thread> mThreads;
	std::condition_variable mEventVar;

	std::mutex mEventMutex;
	bool mStopping = false;

	std::queue<Task> mTasks;

	void start(std::size_t mThreads)
	{
		for (auto i = 0u; i < numThreads; ++i)
		{
			mThreads.emplace_back([=]{
				while (true)
				{
					Task task;
			
					{
						std::unique_lock<std::mutex> lock{mEventMutex};
						mEventVar.wait(lock, [=]{return mStopping || !mTasks.empty();});
						// Now it wakes up, check if it is spurious
						if (mStopping || mTasks.empty())
						{
							break;
						}
						task = std::move(mTasks.front());
						mTasks.pop();
					}

					task();
				}
			})
		}
	}

	void stop() noexcept
	{
		{
			std::unique_lock<std::mutex> lock{mEventMutex};
			mStopping = true;
		}
		mEventVar.notify_all();
		for(auto& thread : mTheads)
		{
			thread.join();
		}
	}
}

int main()
{
	{
		ThreadPool threadPool{36};
		for (auto i = 0; i < 36; ++i)
		{
			pool.enqueue([] {
				auto f = 1000000000;
				while (f > 1)
					f /= 1.00000001;
			});
		}
	}
}