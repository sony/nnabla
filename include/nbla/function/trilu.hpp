// trilu.hpp
#ifndef __NBLA_FUNCTION_TRILU_HPP__
#define __NBLA_FUNCTION_TRILU_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Trilu, int, bool);

template <typename T> class Trilu : public BaseFunction<int, bool> {

protected:
  int k_;
  bool upper_;
  VariablePtr mask_;

public:
  Trilu(const Context &ctx, int k, bool upper)
      : BaseFunction(ctx, k, upper), k_(k), upper_(upper) {}

  virtual ~Trilu() {}
  virtual shared_ptr<Function> copy() const {
    return create_Trilu(ctx_, k_, upper_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Trilu"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual bool grad_depends_output_data(int i, int o) const { return false; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void set_trilu_mask(const Variables &inputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    return false;
  }
};

} // namespace nbla

#endif
