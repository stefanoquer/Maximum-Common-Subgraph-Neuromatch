#include <argp.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "test_utility.h"

// for convenience
using json = nlohmann::json;

#include "graph.h"

using std::cout;
using std::endl;
using std::vector;

static void fail(std::string msg) {
  std::cerr << msg << std::endl;
  exit(1);
}

TestDescription td;
Test test_info;
enum Heuristic { min_max, min_product };

/*******************************************************************************
                             Command-line arguments
*******************************************************************************/

static char doc[] =
    "Find a maximum clique in a graph in DIMACS format\vHEURISTIC can be "
    "min_max or min_product. NODE_HEURISTIC can be degree, total_similarity or norm";
static char args_doc[] = "HEURISTIC NODE_HEURISTIC FILENAME1 FILENAME2 LOAD_FOLDER SAVE_FOLDER";
static struct argp_option options[] = {
    {"quiet", 'q', 0, 0, "Quiet output"},
    {"verbose", 'v', 0, 0, "Verbose output"},
    {"dimacs", 'd', 0, 0, "Read DIMACS format"},
    {"lad", 'l', 0, 0, "Read LAD format"},
    {"connected", 'c', 0, 0, "Solve max common CONNECTED subgraph problem"},
    {"directed", 'i', 0, 0, "Use directed graphs"},
    {"labelled", 'a', 0, 0, "Use edge and vertex labels"},
    {"vertex-labelled-only", 'x', 0, 0,
     "Use vertex labels, but not edge labels"},
    {"big-first", 'b', 0, 0,
     "First try to find an induced subgraph isomorphism, then decrement the "
     "target size"},
    {"timeout", 't', "timeout", 0, "Specify a timeout (seconds)"},
    {"prime", 'p', "prime", 0, "Specify initial incumbent size"},
    {0}};

static struct {
  bool quiet;
  bool verbose;
  bool dimacs;
  bool lad;
  bool connected;
  bool directed;
  bool edge_labelled;
  bool vertex_labelled;
  bool big_first;
  Heuristic heuristic;
  char* filename1;
  char* filename2;
  int timeout;
  int prime;
  int arg_num;
  std::string node_heuristic;
  std::string load_folder;
  std::string save_folder;
} arguments;

static std::atomic<bool> abort_due_to_timeout;

void set_default_arguments() {
  arguments.quiet = false;
  arguments.verbose = false;
  arguments.dimacs = false;
  arguments.lad = false;
  arguments.connected = false;
  arguments.directed = false;
  arguments.edge_labelled = false;
  arguments.vertex_labelled = false;
  arguments.big_first = false;
  arguments.filename1 = NULL;
  arguments.filename2 = NULL;
  arguments.timeout = 0;
  arguments.prime = 0;
  arguments.arg_num = 0;
}

static error_t parse_opt(int key, char* arg, struct argp_state* state) {
  switch (key) {
    case 'd':
      if (arguments.lad)
        fail("The -d and -l options cannot be used together.\n");
      arguments.dimacs = true;
      break;
    case 'l':
      if (arguments.dimacs)
        fail("The -d and -l options cannot be used together.\n");
      arguments.lad = true;
      break;
    case 'q':
      arguments.quiet = true;
      break;
    case 'v':
      arguments.verbose = true;
      break;
    case 'c':
      if (arguments.directed)
        fail("The connected and directed options can't be used together.");
      arguments.connected = true;
      break;
    case 'i':
      if (arguments.connected)
        fail("The connected and directed options can't be used together.");
      arguments.directed = true;
      break;
    case 'a':
      if (arguments.vertex_labelled)
        fail("The -a and -x options can't be used together.");
      arguments.edge_labelled = true;
      arguments.vertex_labelled = true;
      break;
    case 'x':
      if (arguments.edge_labelled)
        fail("The -a and -x options can't be used together.");
      arguments.vertex_labelled = true;
      break;
    case 'b':
      arguments.big_first = true;
      break;
    case 't':
      arguments.timeout = std::stoi(arg);
      break;
    case 'p':
      arguments.prime = std::stoi(arg);
      break;
    case ARGP_KEY_ARG:
      if (arguments.arg_num == 0) {
        if (std::string(arg) == "min_max")
          arguments.heuristic = min_max;
        else if (std::string(arg) == "min_product")
          arguments.heuristic = min_product;
        else
          fail("Unknown heuristic (try min_max or min_product)");
      } else if (arguments.arg_num == 1){
        arguments.node_heuristic = arg;
      }
       else if (arguments.arg_num == 2) {
        arguments.filename1 = arg;
      } else if (arguments.arg_num == 3) {
        arguments.filename2 = arg;
      }
        else if(arguments.arg_num == 4){
          arguments.load_folder = arg;
        }
        else if(arguments.arg_num == 5){
          arguments.save_folder = arg;
        }
       else {
        argp_usage(state);
      }
      arguments.arg_num++;
      break;
    case ARGP_KEY_END:
      if (arguments.arg_num == 0) argp_usage(state);
      break;
    default:
      return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};

/*******************************************************************************
                                     Stats
*******************************************************************************/

unsigned long long nodes{0};

/*******************************************************************************
                                 MCS functions
*******************************************************************************/

struct VtxPair {
  int v;
  int w;
  VtxPair(int v, int w) : v(v), w(w) {}
};

struct Bidomain {
  int l, r;  // start indices of left and right sets
  int left_len, right_len;
  bool is_adjacent;
  Bidomain(int l, int r, int left_len, int right_len, bool is_adjacent)
      : l(l),
        r(r),
        left_len(left_len),
        right_len(right_len),
        is_adjacent(is_adjacent){};
};

void show(const vector<VtxPair>& current, const vector<Bidomain>& domains,
          const vector<int>& left, const vector<int>& right) {
  cout << "Nodes: " << nodes << std::endl;
  cout << "Length of current assignment: " << current.size() << std::endl;
  cout << "Current assignment:";
  for (unsigned int i = 0; i < current.size(); i++) {
    cout << "  (" << current[i].v << " -> " << current[i].w << ")";
  }
  cout << std::endl;
  for (unsigned int i = 0; i < domains.size(); i++) {
    struct Bidomain bd = domains[i];
    cout << "Left  ";
    for (int j = 0; j < bd.left_len; j++) cout << left[bd.l + j] << " ";
    cout << std::endl;
    cout << "Right  ";
    for (int j = 0; j < bd.right_len; j++) cout << right[bd.r + j] << " ";
    cout << std::endl;
  }
  cout << "\n" << std::endl;
}

bool check_sol(const Graph& g0, const Graph& g1,
               const vector<VtxPair>& solution) {
  //return true;
  vector<bool> used_left(g0.n, false);
  vector<bool> used_right(g1.n, false);
  for (unsigned int i = 0; i < solution.size(); i++) {
    struct VtxPair p0 = solution[i];
    if (used_left[p0.v] || used_right[p0.w]) return false;
    used_left[p0.v] = true;
    used_right[p0.w] = true;
    if (g0.label[p0.v] != g1.label[p0.w]) return false;
    for (unsigned int j = i + 1; j < solution.size(); j++) {
      struct VtxPair p1 = solution[j];
      if (g0.adjmat[p0.v][p1.v] != g1.adjmat[p0.w][p1.w]) return false;
    }
  }
  return true;
}

int calc_bound(const vector<Bidomain>& domains) {
  int bound = 0;
  for (const Bidomain& bd : domains) {
    bound += std::min(bd.left_len, bd.right_len);
  }
  return bound;
}

int find_min_value(const vector<int>& arr, int start_idx, int len) {
  int min_v = INT_MAX;
  for (int i = 0; i < len; i++)
    if (arr[start_idx + i] < min_v) min_v = arr[start_idx + i];
  return min_v;
}

int select_bidomain(const vector<Bidomain>& domains, const vector<int>& left,
                    int current_matching_size) {
  // Select the bidomain with the smallest max(leftsize, rightsize), breaking
  // ties on the smallest vertex index in the left set
  int min_size = INT_MAX;
  int min_tie_breaker = INT_MAX;
  int best = -1;
  for (unsigned int i = 0; i < domains.size(); i++) {
    const Bidomain& bd = domains[i];
    if (arguments.connected && current_matching_size > 0 && !bd.is_adjacent)
      continue;
    int len = arguments.heuristic == min_max
                  ? std::max(bd.left_len, bd.right_len)
                  : bd.left_len * bd.right_len;
    if (len < min_size) {
      min_size = len;
      min_tie_breaker = find_min_value(left, bd.l, bd.left_len);
      best = i;
    } else if (len == min_size) {
      int tie_breaker = find_min_value(left, bd.l, bd.left_len);
      if (tie_breaker < min_tie_breaker) {
        min_tie_breaker = tie_breaker;
        best = i;
      }
    }
  }
  return best;
}

// Returns length of left half of array
int partition(vector<int>& all_vv, int start, int len,
              const vector<unsigned int>& adjrow) {
  int i = 0;
  for (int j = 0; j < len; j++) {
    if (adjrow[all_vv[start + j]]) {
      std::swap(all_vv[start + i], all_vv[start + j]);
      i++;
    }
  }
  return i;
}

// multiway is for directed and/or labelled graphs
vector<Bidomain> filter_domains(const vector<Bidomain>& d, vector<int>& left,
                                vector<int>& right, const Graph& g0,
                                const Graph& g1, int v, int w, bool multiway) {
  vector<Bidomain> new_d;
  new_d.reserve(d.size());
  for (const Bidomain& old_bd : d) {
    int l = old_bd.l;
    int r = old_bd.r;
    // After these two partitions, left_len and right_len are the lengths of the
    // arrays of vertices with edges from v or w (int the directed case, edges
    // either from or to v or w)
    int left_len = partition(left, l, old_bd.left_len, g0.adjmat[v]);
    int right_len = partition(right, r, old_bd.right_len, g1.adjmat[w]);
    int left_len_noedge = old_bd.left_len - left_len;
    int right_len_noedge = old_bd.right_len - right_len;
    if (left_len_noedge && right_len_noedge)
      new_d.push_back({l + left_len, r + right_len, left_len_noedge,
                       right_len_noedge, old_bd.is_adjacent});
    if (multiway && left_len && right_len) {
      auto& adjrow_v = g0.adjmat[v];
      auto& adjrow_w = g1.adjmat[w];
      auto l_begin = std::begin(left) + l;
      auto r_begin = std::begin(right) + r;
      std::sort(l_begin, l_begin + left_len,
                [&](int a, int b) { return adjrow_v[a] < adjrow_v[b]; });
      std::sort(r_begin, r_begin + right_len,
                [&](int a, int b) { return adjrow_w[a] < adjrow_w[b]; });
      int l_top = l + left_len;
      int r_top = r + right_len;
      while (l < l_top && r < r_top) {
        unsigned int left_label = adjrow_v[left[l]];
        unsigned int right_label = adjrow_w[right[r]];
        if (left_label < right_label) {
          l++;
        } else if (left_label > right_label) {
          r++;
        } else {
          int lmin = l;
          int rmin = r;
          do {
            l++;
          } while (l < l_top && adjrow_v[left[l]] == left_label);
          do {
            r++;
          } while (r < r_top && adjrow_w[right[r]] == left_label);
          new_d.push_back({lmin, rmin, l - lmin, r - rmin, true});
        }
      }
    } else if (left_len && right_len) {
      new_d.push_back({l, r, left_len, right_len, true});
    }
  }
  return new_d;
}

// returns the index of the smallest value in arr that is >w.
// Assumption: such a value exists
// Assumption: arr contains no duplicates
// Assumption: arr has no values==INT_MAX
int index_of_next_smallest(const vector<int>& arr, int start_idx, int len,
                           int w) {
  int idx = -1;
  int smallest = INT_MAX;
  for (int i = 0; i < len; i++) {
    if (arr[start_idx + i] > w && arr[start_idx + i] < smallest) {
      smallest = arr[start_idx + i];
      idx = i;
    }
  }
  return idx;
}

void remove_vtx_from_left_domain(vector<int>& left, Bidomain& bd, int v) {
  int i = 0;
  while (left[bd.l + i] != v) i++;
  std::swap(left[bd.l + i], left[bd.l + bd.left_len - 1]);
  bd.left_len--;
}

void remove_bidomain(vector<Bidomain>& domains, int idx) {
  domains[idx] = domains[domains.size() - 1];
  domains.pop_back();
}

void solve(const Graph& g0, const Graph& g1, vector<VtxPair>& incumbent,
           vector<VtxPair>& current, vector<Bidomain>& domains,
           vector<int>& left, vector<int>& right,
           unsigned int matching_size_goal) {
  if (abort_due_to_timeout) return;

  if (arguments.verbose) show(current, domains, left, right);
  nodes++;

  if (current.size() > incumbent.size()) {
    incumbent = current;
    if (!arguments.quiet){
      cout << "Incumbent size: " << incumbent.size() << " Recursions: " << nodes << endl;
      test_info.milestones.push_back({incumbent.size(), nodes});
    }
  }

  unsigned int bound = current.size() + calc_bound(domains);
  if (bound <= incumbent.size() || bound < matching_size_goal) return;

  if (arguments.big_first && incumbent.size() == matching_size_goal) return;

  int bd_idx = select_bidomain(domains, left, current.size());
  if (bd_idx == -1)  // In the MCCS case, there may be nothing we can branch on
    return;
  Bidomain& bd = domains[bd_idx];

  int v = find_min_value(left, bd.l, bd.left_len);
  remove_vtx_from_left_domain(left, domains[bd_idx], v);

  // Try assigning v to each vertex w in the colour class beginning at bd.r, in
  // turn
  int w = -1;
  bd.right_len--;
  for (int i = 0; i <= bd.right_len; i++) {
    int idx = index_of_next_smallest(right, bd.r, bd.right_len + 1, w);
    w = right[bd.r + idx];

    // swap w to the end of its colour class
    right[bd.r + idx] = right[bd.r + bd.right_len];
    right[bd.r + bd.right_len] = w;

    auto new_domains =
        filter_domains(domains, left, right, g0, g1, v, w,
                       arguments.directed || arguments.edge_labelled);
    current.push_back(VtxPair(v, w));
    solve(g0, g1, incumbent, current, new_domains, left, right,
          matching_size_goal);
    current.pop_back();
  }
  bd.right_len++;
  if (bd.left_len == 0) remove_bidomain(domains, bd_idx);
  solve(g0, g1, incumbent, current, domains, left, right, matching_size_goal);
}

vector<VtxPair> mcs(const Graph& g0, const Graph& g1) {
  vector<int> left;   // the buffer of vertex indices for the left partitions
  vector<int> right;  // the buffer of vertex indices for the right partitions

  auto domains = vector<Bidomain>{};

  std::set<unsigned int> left_labels;
  std::set<unsigned int> right_labels;
  for (unsigned int label : g0.label) left_labels.insert(label);
  for (unsigned int label : g1.label) right_labels.insert(label);
  std::set<unsigned int> labels;  // labels that appear in both graphs
  std::set_intersection(std::begin(left_labels), std::end(left_labels),
                        std::begin(right_labels), std::end(right_labels),
                        std::inserter(labels, std::begin(labels)));

  // Create a bidomain for each label that appears in both graphs
  for (unsigned int label : labels) {
    int start_l = left.size();
    int start_r = right.size();

    for (int i = 0; i < g0.n; i++)
      if (g0.label[i] == label) left.push_back(i);
    for (int i = 0; i < g1.n; i++)
      if (g1.label[i] == label) right.push_back(i);

    int left_len = left.size() - start_l;
    int right_len = right.size() - start_r;
    domains.push_back({start_l, start_r, left_len, right_len, false});
  }

  vector<VtxPair> incumbent(arguments.prime, {-1, -1});

  if (arguments.big_first) {
    for (int k = 0; k < g0.n; k++) {
      unsigned int goal = g0.n - k;
      auto left_copy = left;
      auto right_copy = right;
      auto domains_copy = domains;
      vector<VtxPair> current;
      solve(g0, g1, incumbent, current, domains_copy, left_copy, right_copy,
            goal);
      if (incumbent.size() == goal || abort_due_to_timeout) break;
      if (!arguments.quiet) cout << "Upper bound: " << goal - 1 << std::endl;
    }

  } else {
    vector<VtxPair> current;
    solve(g0, g1, incumbent, current, domains, left, right, 1);
  }

  test_info.recursions = nodes;
  return incumbent;
}

vector<float> calculate_degrees(const Graph& g) {
  vector<int> degree(g.n, 0);
  for (int v = 0; v < g.n; v++) {
    for (int w = 0; w < g.n; w++) {
      unsigned int mask = 0xFFFFu;
      if (g.adjmat[v][w] & mask) degree[v]++;
      if (g.adjmat[v][w] & ~mask) degree[v]++;  // inward edge, in directed case
    }
  }
  std::vector<float> f_degree(degree.begin(), degree.end());
  return f_degree;
}

int sum(const vector<float>& vec) {
  return std::accumulate(std::begin(vec), std::end(vec), 0);
}


int main(int argc, char** argv) {
  set_default_arguments();
  argp_parse(&argp, argc, argv, 0, 0, 0);
  struct timespec start, finish;
  double time_elapsed;
  char format = arguments.dimacs ? 'D' : arguments.lad ? 'L' : 'B';
  struct Graph g0 =
      readGraph(arguments.filename1, format, arguments.directed,
                arguments.edge_labelled, arguments.vertex_labelled);
  struct Graph g1 =
      readGraph(arguments.filename2, format, arguments.directed,
                arguments.edge_labelled, arguments.vertex_labelled);


  //**TEST STRUCTURE**//
  std::string folder_name(arguments.filename1);
  folder_name.erase(folder_name.size()-4, 2);
  std::stringstream ss(folder_name);
  std::vector<std::string> v;
  std::string buff;
  while(std::getline(ss, buff, '/')){
    v.push_back(buff);
  }
  std::string test_name = v[v.size()-1];

  td = TestDescription(test_name, arguments.vertex_labelled, arguments.directed, arguments.connected, arguments.edge_labelled, arguments.timeout, arguments.node_heuristic);
  test_info = Test(td);

 //**PRECOMPUTATION LOADING**//
  std::string node_heuristic(arguments.node_heuristic);
  vector<float> v1;
  vector<float> v2;
  if(node_heuristic != "classic"){
    std::string load_folder(arguments.load_folder);
    std::ifstream jsonfile1(load_folder + "/" + test_name + "/g1.json");
    json mylist1;
    jsonfile1 >> mylist1;

    std::ifstream jsonfile2(load_folder + "/" + test_name+ "/g2.json");
    json mylist2;
    jsonfile2 >> mylist2;

    for(auto& elem: mylist1){
      v1.push_back(elem.get<float>());
    }
    
    for(auto& elem: mylist2){
      v2.push_back(elem.get<float>());
    }
  }
  //**END PRECOMPUTATION LOADING**//

  std::thread timeout_thread;
  std::mutex timeout_mutex;
  std::condition_variable timeout_cv;
  abort_due_to_timeout.store(false);
  bool aborted = false;

  if (0 != arguments.timeout) {
    timeout_thread = std::thread([&] {
      auto abort_time = std::chrono::steady_clock::now() +
                        std::chrono::seconds(arguments.timeout);
      {
        /* Sleep until either we've reached the time limit,
         * or we've finished all the work. */
        std::unique_lock<std::mutex> guard(timeout_mutex);
        while (!abort_due_to_timeout.load()) {
          if (std::cv_status::timeout ==
              timeout_cv.wait_until(guard, abort_time)) {
            /* We've woken up, and it's due to a timeout. */
            aborted = true;
            break;
          }
        }
      }
      abort_due_to_timeout.store(true);
    });
  }

  // auto start = std::chrono::steady_clock::now();

vector<float> g0_deg;
vector<float> g1_deg;
if(node_heuristic != "classic"){
  g0_deg = v1;
  g1_deg = v2;
}
else{
  g0_deg = calculate_degrees(g0);
  g1_deg = calculate_degrees(g1);
}

  // As implemented here, g1_dense and g0_dense are false for all instances
  // in the Experimental Evaluation section of the paper.  Thus,
  // we always sort the vertices in descending order of degree (or total degree,
  // in the case of directed graphs.  Improvements could be made here: it would
  // be nice if the program explored exactly the same search tree if both
  // input graphs were complemented.
  vector<int> vv0(g0.n);
  std::iota(std::begin(vv0), std::end(vv0), 0);
  bool g1_dense = sum(g1_deg) > g1.n * (g1.n - 1);
  std::stable_sort(std::begin(vv0), std::end(vv0), [&](int a, int b) {
    return g1_dense ? (g0_deg[a] < g0_deg[b]) : (g0_deg[a] > g0_deg[b]);
  });
  vector<int> vv1(g1.n);
  std::iota(std::begin(vv1), std::end(vv1), 0);
  bool g0_dense = sum(g0_deg) > g0.n * (g0.n - 1);
  std::stable_sort(std::begin(vv1), std::end(vv1), [&](int a, int b) {
    return g0_dense ? (g1_deg[a] < g1_deg[b]) : (g1_deg[a] > g1_deg[b]);
  });
  
  struct Graph g0_sorted = induced_subgraph(g0, vv0);
  struct Graph g1_sorted = induced_subgraph(g1, vv1);

  clock_gettime(CLOCK_MONOTONIC, &start);

  vector<VtxPair> solution = mcs(g0_sorted, g1_sorted);

  clock_gettime(CLOCK_MONOTONIC, &finish);

  std::vector<std::pair<int,int>> solution_to_save;
  // Convert to indices from original, unsorted graphs
  for (auto& vtx_pair : solution) {
    vtx_pair.v = vv0[vtx_pair.v];
    vtx_pair.w = vv1[vtx_pair.w];
    solution_to_save.push_back({vtx_pair.v, vtx_pair.w});
  }

  test_info.solution = solution_to_save;

  time_elapsed = (finish.tv_sec - start.tv_sec);  // calculating elapsed seconds
  time_elapsed += (double)(finish.tv_nsec - start.tv_nsec) /
                  1000000000.0;  // adding elapsed nanoseconds
  test_info.total_time = time_elapsed;
  // auto stop = std::chrono::steady_clock::now();
  // auto time_elapsed =
  // std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

  /* Clean up the timeout thread */
  if (timeout_thread.joinable()) {
    {
      std::unique_lock<std::mutex> guard(timeout_mutex);
      abort_due_to_timeout.store(true);
      timeout_cv.notify_all();
    }
    timeout_thread.join();
  }

  if (!check_sol(g0, g1, solution)) fail("*** Error: Invalid solution\n");

  cout << "Solution size " << solution.size() << std::endl;
  for (int i = 0; i < g0.n; i++)
    for (unsigned int j = 0; j < solution.size(); j++)
      if (solution[j].v == i){
        cout << "(" << solution[j].v << " -> " << solution[j].w << ") ";
      }
  cout << std::endl;

  printf(">>> %lu -  %015.10f\n", solution.size(), time_elapsed);

  cout << "Nodes:                      " << nodes << endl;
  cout << "CPU time (ms):              " << time_elapsed << endl;
  if (aborted) cout << "TIMEOUT" << endl;

  test_info.to_string();
  std::string save_folder(arguments.save_folder);
  save_json(test_info, save_folder);
}
