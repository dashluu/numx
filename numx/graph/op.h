#pragma once

#include "../core/array_iter.h"

namespace nx::graph {
    using namespace nx::utils;
    using namespace nx::core;

    enum struct Opcode {
        NOP,
        EMPTY,
        RANDN,
        ARANGE,
        FULL,
        ADD,
        SUB,
        MUL,
        DIV,
        EQ,
        NEQ,
        GT,
        GEQ,
        LT,
        LEQ,
        MINIMUM,
        MAXIMUM,
        MATMUL,
        SQ,
        SQRT,
        NEG,
        COPY,
        EXP,
        LOG,
        RECIP,
        RESHAPE,
        PERMUTE,
        BROADCAST,
        SQUEEZE,
        UNSQUEEZE,
        SLICE,
        SUM,
        MAX,
        MIN,
        ARGMAX,
        ARGMIN,
        ASTYPE,
        // Used to get the number of enums
        COUNT
    };

    enum struct Optype {
        INITIALIZER,
        UNARY,
        BINARY,
        TRANSFORM,
        REDUCE
    };

    enum struct BinaryMode {
        ELMWISE,
        CMP,
        MATMUL
    };

    struct Op : public std::enable_shared_from_this<Op> {
        using OpPtr = std::shared_ptr<Op>;

    protected:
        ArrayData m_data;
        OpPtr m_partial_grad, m_grad;
        // Note: m_grad_enabled cannot be used to set gradient flow once the computational graph is compiled or forwarded
        bool m_grad_enabled;

        OpPtr nonconst_from_this() const { return std::const_pointer_cast<Op>(std::static_pointer_cast<const Op>(shared_from_this())); }
        OpPtr detach_this() const;

    public:
        Op(const ArrayData &data) : m_data(data) { m_grad_enabled = m_data.get_dtype()->is_float(); }
        Op(const Op &) = delete;
        virtual ~Op() = default;
        Op &operator=(const Op &) = delete;
        virtual Opcode get_opcode() const = 0;
        virtual Optype get_optype() const = 0;
        virtual const std::string &get_opname() const = 0;
        ArrayData &get_data() { return m_data; }
        OpPtr get_grad() const { return m_grad; }
        OpPtr get_partial_grad() const { return m_partial_grad; }
        bool is_grad_enabled() const { return m_grad_enabled; }

        virtual void enable_grad(bool enabled) {
            if (!m_data.get_dtype()->is_float() && enabled) {
                throw std::runtime_error(std::format("Only floating-point arrays can have gradients but array {} has type {}.", m_data.get_id().str(), m_data.get_dtype()->str()));
            }

            m_grad_enabled = enabled;
        }

        void zero_grad();
        void one_grad();
        void iadd_grad(OpPtr grad);
        void isub_grad(OpPtr grad);
        void slice_grad(OpPtr grad, const RangeVec &ranges);
        virtual void grad_fn() const {}
        virtual const std::string str() const { return std::format("opname: {}", get_opname()); }
    };

    using OpPtr = std::shared_ptr<Op>;

    struct InitializerOp : public Op {
    public:
        InitializerOp(const ArrayData &data) : Op(data) {}
        Optype get_optype() const override { return Optype::INITIALIZER; }
    };

    struct Nop : public InitializerOp {
    public:
        inline static const std::string s_opname = "nop";
        Nop(const ArrayData &data) : InitializerOp(data) {}
        Opcode get_opcode() const override { return Opcode::NOP; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct EmptyOp : public InitializerOp {
    public:
        inline static const std::string s_opname = "empty";
        EmptyOp(const ArrayData &data) : InitializerOp(data) {}
        Opcode get_opcode() const override { return Opcode::EMPTY; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct ArangeOp : public InitializerOp {
    private:
        ShapeView m_view;
        isize m_start;
        isize m_step;

    public:
        inline static const std::string s_opname = "arange";
        ArangeOp(const ArrayData &data, const ShapeView &view, isize start, isize step) : InitializerOp(data), m_view(view), m_start(start), m_step(step) {}
        Opcode get_opcode() const override { return Opcode::ARANGE; }
        const std::string &get_opname() const override { return s_opname; }
        const ShapeView &get_view() const { return m_view; }
        isize get_start() const { return m_start; }
        isize get_step() const { return m_step; }
        const std::string str() const override { return std::format("{}, view: {}, start: {}, step: {}", InitializerOp::str(), join_nums(m_view), m_start, m_step); }
    };

    struct FullOp : public InitializerOp {
    private:
        ShapeView m_view;
        isize m_const;

    public:
        inline static const std::string s_opname = "full";
        FullOp(const ArrayData &data, const ShapeView &view, isize constant) : InitializerOp(data), m_view(view), m_const(constant) {}
        Opcode get_opcode() const override { return Opcode::FULL; }
        const std::string &get_opname() const override { return s_opname; }
        const ShapeView &get_view() const { return m_view; }
        isize get_const() const { return m_const; }
        const std::string str() const override { return std::format("{}, view: {}, value: {}", InitializerOp::str(), join_nums(m_view), m_data.get_dtype()->value_str(m_const)); }
    };

    struct UnaryOp : public Op {
    protected:
        bool m_in_place;
        OpPtr m_operand;

    public:
        UnaryOp(const ArrayData &data, OpPtr operand, bool in_place) : Op(data), m_operand(operand), m_in_place(in_place) {}
        Optype get_optype() const override { return Optype::UNARY; }
        OpPtr get_operand() { return m_operand; }
        bool is_in_place() const { return m_in_place; }
        const std::string str() const override { return std::format("{}, in-place: {}, operand: {}", Op::str(), m_in_place, m_operand->get_data().get_id().str()); }
    };

    struct BinaryOp : public Op {
    protected:
        OpPtr m_lhs;
        OpPtr m_rhs;
        BinaryMode m_mode;

    public:
        BinaryOp(const ArrayData &data, OpPtr lhs, OpPtr rhs, BinaryMode mode) : Op(data), m_lhs(lhs), m_rhs(rhs), m_mode(mode) {}
        Optype get_optype() const override { return Optype::BINARY; }
        OpPtr get_lhs() { return m_lhs; }
        OpPtr get_rhs() { return m_rhs; }
        BinaryMode get_mode() const { return m_mode; }
        const std::string str() const override { return std::format("{}, lhs: {}, rhs: {}", Op::str(), m_lhs->get_data().get_id().str(), m_rhs->get_data().get_id().str()); }
    };

    struct ElmwiseBinaryOp : public BinaryOp {
    protected:
        bool m_in_place;

    public:
        ElmwiseBinaryOp(const ArrayData &data, OpPtr lhs, OpPtr rhs, bool in_place) : BinaryOp(data, lhs, rhs, BinaryMode::ELMWISE), m_in_place(in_place) {}
        bool is_in_place() const { return m_in_place; }
        const std::string str() const override { return std::format("{}, in-place: {}, lhs: {}, rhs: {}", Op::str(), m_in_place, m_lhs->get_data().get_id().str(), m_rhs->get_data().get_id().str()); }
    };

    struct CmpOp : public BinaryOp {
    public:
        CmpOp(const ArrayData &data, OpPtr lhs, OpPtr rhs) : BinaryOp(data, lhs, rhs, BinaryMode::CMP) {}
    };

    struct TransformOp : public Op {
    protected:
        OpPtr m_operand;

    public:
        TransformOp(const ArrayData &data, OpPtr operand) : Op(data), m_operand(operand) {}
        Optype get_optype() const override { return Optype::TRANSFORM; }
        OpPtr get_operand() { return m_operand; }
        const std::string str() const override { return std::format("{}, operand: {}", Op::str(), m_operand->get_data().get_id().str()); }
    };

    struct ReduceOp : public Op {
    protected:
        OpPtr m_operand;
        ShapeDims m_remaining_dims;
        ShapeDims m_reduce_dims;

    public:
        ReduceOp(const ArrayData &data, OpPtr operand, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) : Op(data), m_operand(operand), m_remaining_dims(remaining_dims), m_reduce_dims(reduce_dims) {}
        Optype get_optype() const override { return Optype::REDUCE; }
        OpPtr get_operand() { return m_operand; }
        const ShapeDims &get_remaining_dims() const { return m_remaining_dims; }
        const ShapeDims &get_reduce_dims() const { return m_reduce_dims; }
        const std::string str() const override { return std::format("{}, operand: {}, kept dims: {}, reduce dims: {}", Op::str(), m_operand->get_data().get_id().str(), join_nums(m_remaining_dims), join_nums(m_reduce_dims)); }
    };

    struct AddOp : public ElmwiseBinaryOp {
    public:
        inline static const std::string s_opname = "add";
        AddOp(const ArrayData &data, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(data, lhs, rhs, in_place) {}
        Opcode get_opcode() const override { return Opcode::ADD; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct SubOp : public ElmwiseBinaryOp {
    public:
        inline static const std::string s_opname = "sub";
        SubOp(const ArrayData &data, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(data, lhs, rhs, in_place) {}
        Opcode get_opcode() const override { return Opcode::SUB; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct MulOp : public ElmwiseBinaryOp {
    public:
        inline static const std::string s_opname = "mul";
        MulOp(const ArrayData &data, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(data, lhs, rhs, in_place) {}
        Opcode get_opcode() const override { return Opcode::MUL; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct DivOp : public ElmwiseBinaryOp {
    public:
        inline static const std::string s_opname = "div";
        DivOp(const ArrayData &data, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(data, lhs, rhs, in_place) {}
        Opcode get_opcode() const override { return Opcode::DIV; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct EqOp : public CmpOp {
    public:
        inline static const std::string s_opname = "eq";
        EqOp(const ArrayData &data, OpPtr lhs, OpPtr rhs) : CmpOp(data, lhs, rhs) {}
        Opcode get_opcode() const override { return Opcode::EQ; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct NeqOp : public CmpOp {
    public:
        inline static const std::string s_opname = "neq";
        NeqOp(const ArrayData &data, OpPtr lhs, OpPtr rhs) : CmpOp(data, lhs, rhs) {}
        Opcode get_opcode() const override { return Opcode::NEQ; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct LtOp : public CmpOp {
    public:
        inline static const std::string s_opname = "lt";
        LtOp(const ArrayData &data, OpPtr lhs, OpPtr rhs) : CmpOp(data, lhs, rhs) {}
        Opcode get_opcode() const override { return Opcode::LT; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct GtOp : public CmpOp {
    public:
        inline static const std::string s_opname = "gt";
        GtOp(const ArrayData &data, OpPtr lhs, OpPtr rhs) : CmpOp(data, lhs, rhs) {}
        Opcode get_opcode() const override { return Opcode::GT; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct LeqOp : public CmpOp {
    public:
        inline static const std::string s_opname = "leq";
        LeqOp(const ArrayData &data, OpPtr lhs, OpPtr rhs) : CmpOp(data, lhs, rhs) {}
        Opcode get_opcode() const override { return Opcode::LEQ; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct GeqOp : public CmpOp {
    public:
        inline static const std::string s_opname = "geq";
        GeqOp(const ArrayData &data, OpPtr lhs, OpPtr rhs) : CmpOp(data, lhs, rhs) {}
        Opcode get_opcode() const override { return Opcode::GEQ; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct MinimumOp : public ElmwiseBinaryOp {
    public:
        inline static const std::string s_opname = "minimum";
        MinimumOp(const ArrayData &data, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(data, lhs, rhs, in_place) {}
        Opcode get_opcode() const override { return Opcode::MINIMUM; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct MaximumOp : public ElmwiseBinaryOp {
    public:
        inline static const std::string s_opname = "maximum";
        MaximumOp(const ArrayData &data, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(data, lhs, rhs, in_place) {}
        Opcode get_opcode() const override { return Opcode::MAXIMUM; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct MatmulOp : public BinaryOp {
    public:
        inline static const std::string s_opname = "matmul";
        MatmulOp(const ArrayData &data, OpPtr lhs, OpPtr rhs) : BinaryOp(data, lhs, rhs, BinaryMode::MATMUL) {}
        Opcode get_opcode() const override { return Opcode::MATMUL; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct SqOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "sq";
        SqOp(const ArrayData &data, OpPtr operand, bool in_place) : UnaryOp(data, operand, in_place) {}
        Opcode get_opcode() const override { return Opcode::SQ; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct SqrtOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "sqrt";
        SqrtOp(const ArrayData &data, OpPtr operand, bool in_place) : UnaryOp(data, operand, in_place) {}
        Opcode get_opcode() const override { return Opcode::SQRT; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct NegOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "neg";
        NegOp(const ArrayData &data, OpPtr operand, bool in_place) : UnaryOp(data, operand, in_place) {}
        Opcode get_opcode() const override { return Opcode::NEG; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct CopyOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "copy";
        CopyOp(const ArrayData &data, OpPtr operand) : UnaryOp(data, operand, false) {}
        Opcode get_opcode() const override { return Opcode::COPY; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct ExpOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "exp";
        ExpOp(const ArrayData &data, OpPtr operand, bool in_place) : UnaryOp(data, operand, in_place) {}
        Opcode get_opcode() const override { return Opcode::EXP; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct LogOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "log";
        LogOp(const ArrayData &data, OpPtr operand, bool in_place) : UnaryOp(data, operand, in_place) {}
        Opcode get_opcode() const override { return Opcode::LOG; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct RecipOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "recip";
        RecipOp(const ArrayData &data, OpPtr operand, bool in_place) : UnaryOp(data, operand, in_place) {}
        Opcode get_opcode() const override { return Opcode::RECIP; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct ReshapeOp : public TransformOp {
    private:
        ShapeView m_view;

    public:
        inline static const std::string s_opname = "reshape";
        ReshapeOp(const ArrayData &data, OpPtr operand, const ShapeView &view) : TransformOp(data, operand), m_view(view) {}
        const ShapeView &get_view() const { return m_view; }
        Opcode get_opcode() const override { return Opcode::RESHAPE; }
        const std::string &get_opname() const override { return s_opname; }
        const std::string str() const override { return std::format("{}, view: ({})", TransformOp::str(), join_nums(m_view)); }
        void grad_fn() const override;
    };

    struct SliceOp : public TransformOp {
    private:
        std::vector<Range> m_ranges;

    public:
        inline static const std::string s_opname = "slice";
        SliceOp(const ArrayData &data, OpPtr operand, const std::vector<Range> &ranges) : TransformOp(data, operand), m_ranges(ranges) {}
        const std::vector<Range> &get_ranges() const { return m_ranges; }
        Opcode get_opcode() const override { return Opcode::SLICE; }
        const std::string &get_opname() const override { return s_opname; }

        const std::string str() const override {
            return std::format("{}, ranges: ({})", TransformOp::str(), join<Range>(m_ranges, [](Range range) { return range.str(); }));
        }

        void grad_fn() const override;
    };

    struct PermuteOp : public TransformOp {
    private:
        ShapeDims m_dims;

    public:
        inline static const std::string s_opname = "permute";
        PermuteOp(const ArrayData &data, OpPtr operand, const ShapeDims &dims) : TransformOp(data, operand), m_dims(dims) {}
        const ShapeDims &get_perm() const { return m_dims; }
        Opcode get_opcode() const override { return Opcode::PERMUTE; }
        const std::string &get_opname() const override { return s_opname; }
        const std::string str() const override { return std::format("{}, permutation: ({})", TransformOp::str(), join_nums(m_dims)); }
        void grad_fn() const override;
    };

    struct BroadcastOp : public TransformOp {
    private:
        ShapeView m_input_view;
        ShapeView m_output_view;
        ShapeDims m_dims;

    public:
        inline static const std::string s_opname = "broadcast";
        BroadcastOp(const ArrayData &data, OpPtr operand, const ShapeView &input_view, const ShapeView &output_view, const ShapeDims &dims) : TransformOp(data, operand), m_input_view(input_view), m_output_view(output_view), m_dims(dims) {}
        const ShapeView &get_input_view() const { return m_input_view; }
        const ShapeView &get_output_view() const { return m_output_view; }
        const ShapeDims &get_dims() const { return m_dims; }
        Opcode get_opcode() const override { return Opcode::BROADCAST; }
        const std::string &get_opname() const override { return s_opname; }
        const std::string str() const override { return std::format("{}, input view: ({}), output view: ({})", TransformOp::str(), join_nums(m_input_view), join_nums(m_output_view)); }
        void grad_fn() const override;
    };

    struct SqueezeOp : public TransformOp {
    private:
        ShapeDims m_dims;

    public:
        inline static const std::string s_opname = "squeeze";
        SqueezeOp(const ArrayData &data, OpPtr operand, const ShapeDims &dims) : TransformOp(data, operand), m_dims(dims) {}
        const ShapeDims &get_dims() const { return m_dims; }
        Opcode get_opcode() const override { return Opcode::SQUEEZE; }
        const std::string &get_opname() const override { return s_opname; }
        const std::string str() const override { return std::format("{}, dims: ({})", TransformOp::str(), join_nums(m_dims)); }
        void grad_fn() const override;
    };

    struct UnsqueezeOp : public TransformOp {
    private:
        ShapeDims m_dims;

    public:
        inline static const std::string s_opname = "unsqueeze";
        UnsqueezeOp(const ArrayData &data, OpPtr operand, const ShapeDims &dims) : TransformOp(data, operand), m_dims(dims) {}
        const ShapeDims &get_dims() const { return m_dims; }
        Opcode get_opcode() const override { return Opcode::UNSQUEEZE; }
        const std::string &get_opname() const override { return s_opname; }
        const std::string str() const override { return std::format("{}, dims: ({})", TransformOp::str(), join_nums(m_dims)); }
        void grad_fn() const override;
    };

    struct AstypeOp : public TransformOp {
    private:
        DtypePtr m_dtype;

    public:
        inline static const std::string s_opname = "astype";
        AstypeOp(const ArrayData &data, OpPtr operand, DtypePtr dtype) : TransformOp(data, operand), m_dtype(dtype) {}
        DtypePtr get_dtype() const { return m_dtype; }
        Opcode get_opcode() const override { return Opcode::ASTYPE; }
        const std::string &get_opname() const override { return s_opname; }
        const std::string str() const override { return std::format("{}, dtype: {}", TransformOp::str(), m_dtype->str()); }
    };

    struct SumOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "sum";
        SumOp(const ArrayData &data, OpPtr operand, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) : ReduceOp(data, operand, remaining_dims, reduce_dims) {}
        Opcode get_opcode() const override { return Opcode::SUM; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct MaxOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "max";
        MaxOp(const ArrayData &data, OpPtr operand, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) : ReduceOp(data, operand, remaining_dims, reduce_dims) {}
        Opcode get_opcode() const override { return Opcode::MAX; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct MinOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "min";
        MinOp(const ArrayData &data, OpPtr operand, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) : ReduceOp(data, operand, remaining_dims, reduce_dims) {}
        Opcode get_opcode() const override { return Opcode::MIN; }
        const std::string &get_opname() const override { return s_opname; }
        void grad_fn() const override;
    };

    struct ArgmaxOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "argmax";
        ArgmaxOp(const ArrayData &data, OpPtr operand, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) : ReduceOp(data, operand, remaining_dims, reduce_dims) {}
        Opcode get_opcode() const override { return Opcode::ARGMAX; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct ArgminOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "argmin";
        ArgminOp(const ArrayData &data, OpPtr operand, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) : ReduceOp(data, operand, remaining_dims, reduce_dims) {}
        Opcode get_opcode() const override { return Opcode::ARGMIN; }
        const std::string &get_opname() const override { return s_opname; }
    };
} // namespace nx::graph