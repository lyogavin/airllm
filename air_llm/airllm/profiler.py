



class LayeredProfiler:
    def __init__(self):
        self.profiling_time_dict = {}


    def add_profiling_time(self, item, time):

        if not item in self.profiling_time_dict:
            self.profiling_time_dict[item] = []

        self.profiling_time_dict[item].append(time)


    def clear_profiling_time(self):
        for item in self.profiling_time_dict.keys():
            self.profiling_time_dict[item] = []

    def print_profiling_time(self):
        for item in self.profiling_time_dict.keys():
            print(f"total time for {item}: {sum(self.profiling_time_dict[item])}")

