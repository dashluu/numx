#pragma once

#include "array_descriptor.h"

namespace nx::core {
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
    protected:
        // Note: m_grad_enabled cannot be used to set gradient flow once the computational graph is compiled or forwarded
        bool m_grad_enabled;
        ArrayDescriptor m_descriptor;

    public:
        Op(const ArrayDescriptor &descriptor) : m_descriptor(descriptor) { m_grad_enabled = descriptor.get_dtype()->is_float(); }
        Op(const Op &) = delete;
        virtual ~Op() = default;
        Op &operator=(const Op &) = delete;
        virtual Opcode get_opcode() const = 0;
        virtual const std::string &get_opname() const = 0;
        virtual Optype get_optype() const = 0;
        const ArrayDescriptor &get_descriptor() const { return m_descriptor; }
        bool is_grad_enabled() const { return m_grad_enabled; }

        // Leave this as virtual, some ops need to disable enabling grad since they are not differentiable
        virtual void enable_grad(bool enabled) {
            if (!m_descriptor.get_dtype()->is_float() && enabled) {
                throw std::runtime_error(std::format("Only floating-point arrays can have gradients but array {} has type {}.", m_descriptor.get_id().str(), m_descriptor.get_dtype()->str()));
            }
            m_grad_enabled = enabled;
        }

        virtual const std::string str() const { return std::format("opname: {}", get_opname()); }
    };

    using OpPtr = std::shared_ptr<Op>;

    struct InitializerOp : public Op {
    public:
        InitializerOp(const ArrayDescriptor &descriptor) : Op(descriptor) {}
        Optype get_optype() const override { return Optype::INITIALIZER; }
    };

    struct Nop : public InitializerOp {
    public:
        inline static const std::string s_opname = "nop";
        Nop(const ArrayDescriptor &descriptor) : InitializerOp(descriptor) {}
        Opcode get_opcode() const override { return Opcode::NOP; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct EmptyOp : public InitializerOp {
    public:
        inline static const std::string s_opname = "empty";
        EmptyOp(const ArrayDescriptor &descriptor) : InitializerOp(descriptor) {}
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
        ArangeOp(const ArrayDescriptor &descriptor, const ShapeView &view, isize start, isize step) : InitializerOp(descriptor), m_view(view), m_start(start), m_step(step) {}
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
        FullOp(const ArrayDescriptor &descriptor, const ShapeView &view, isize constant, DtypePtr dtype) : InitializerOp(descriptor), m_view(view), m_const(constant) {}
        Opcode get_opcode() const override { return Opcode::FULL; }
        const std::string &get_opname() const override { return s_opname; }
        const ShapeView &get_view() const { return m_view; }
        isize get_const() const { return m_const; }
        const std::string str() const override { return std::format("{}, view: {}, value: {}", InitializerOp::str(), join_nums(m_view), m_descriptor.get_dtype()->value_str(m_const)); }
    };

    struct UnaryOp : public Op {
    protected:
        bool m_in_place;
        OpPtr m_operand;

    public:
        UnaryOp(const ArrayDescriptor &descriptor, OpPtr operand, bool in_place) : Op(descriptor), m_operand(operand), m_in_place(in_place) {}
        Optype get_optype() const override { return Optype::UNARY; }
        OpPtr get_operand() const { return m_operand; }
        bool is_in_place() const { return m_in_place; }
        const std::string str() const override { return std::format("{}, operand: {}, in-place: {}", Op::str(), m_operand->get_descriptor().get_id().str(), m_in_place); }
    };

    struct BinaryOp : public Op {
    protected:
        OpPtr m_lhs;
        OpPtr m_rhs;

    public:
        BinaryOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs) : Op(descriptor), m_lhs(lhs), m_rhs(rhs) {}
        Optype get_optype() const override { return Optype::BINARY; }
        virtual BinaryMode get_mode() const = 0;
        OpPtr get_lhs() const { return m_lhs; }
        OpPtr get_rhs() const { return m_rhs; }
        const std::string str() const override { return std::format("{}, lhs: {}, rhs: {}", Op::str(), m_lhs->get_descriptor().get_id().str(), m_rhs->get_descriptor().get_id().str()); }
    };

    struct ElmwiseBinaryOp : public BinaryOp {
    protected:
        bool m_in_place;

    public:
        ElmwiseBinaryOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : BinaryOp(descriptor, lhs, rhs), m_in_place(in_place) {}
        BinaryMode get_mode() const override { return BinaryMode::ELMWISE; }
        bool is_in_place() const { return m_in_place; }
        const std::string str() const override { return std::format("{}, in-place: {}", BinaryOp::str(), m_in_place); }
    };

    struct CmpOp : public BinaryOp {
    public:
        CmpOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs) : BinaryOp(descriptor, lhs, rhs) {}
        BinaryMode get_mode() const override { return BinaryMode::CMP; }
        void enable_grad(bool enabled) override { m_grad_enabled = false; }
    };

    struct TransformOp : public Op {
    protected:
        OpPtr m_operand;

    public:
        TransformOp(const ArrayDescriptor &descriptor, OpPtr operand) : Op(descriptor), m_operand(operand) {}
        Optype get_optype() const override { return Optype::TRANSFORM; }
        OpPtr get_operand() const { return m_operand; }
        const std::string str() const override { return std::format("{}, operand: {}", Op::str(), m_operand->get_descriptor().get_id().str()); }
    };

    struct ReduceOp : public Op {
    protected:
        OpPtr m_operand;
        ShapeDims m_remaining_dims;
        ShapeDims m_reduce_dims;

    public:
        ReduceOp(const ArrayDescriptor &descriptor, OpPtr operand, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) : Op(descriptor), m_operand(operand), m_remaining_dims(remaining_dims), m_reduce_dims(reduce_dims) {}
        Optype get_optype() const override { return Optype::REDUCE; }
        OpPtr get_operand() const { return m_operand; }
        const ShapeDims &get_remaining_dims() const { return m_remaining_dims; }
        const ShapeDims &get_reduce_dims() const { return m_reduce_dims; }
        const std::string str() const override { return std::format("{}, operand: {}, kept dims: {}, reduce dims: {}", Op::str(), m_operand->get_descriptor().get_id().str(), join_nums(m_remaining_dims), join_nums(m_reduce_dims)); }
    };

    struct AddOp : public ElmwiseBinaryOp {
    public:
        inline static const std::string s_opname = "add";
        AddOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(descriptor, lhs, rhs, in_place) {}
        Opcode get_opcode() const override { return Opcode::ADD; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct SubOp : public ElmwiseBinaryOp {
    public:
        inline static const std::string s_opname = "sub";
        SubOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(descriptor, lhs, rhs, in_place) {}
        Opcode get_opcode() const override { return Opcode::SUB; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct MulOp : public ElmwiseBinaryOp {
    public:
        inline static const std::string s_opname = "mul";
        MulOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(descriptor, lhs, rhs, in_place) {}
        Opcode get_opcode() const override { return Opcode::MUL; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct DivOp : public ElmwiseBinaryOp {
    public:
        inline static const std::string s_opname = "div";
        DivOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(descriptor, lhs, rhs, in_place) {}
        Opcode get_opcode() const override { return Opcode::DIV; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct EqOp : public CmpOp {
    public:
        inline static const std::string s_opname = "eq";
        EqOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs) : CmpOp(descriptor, lhs, rhs) {}
        Opcode get_opcode() const override { return Opcode::EQ; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct NeqOp : public CmpOp {
    public:
        inline static const std::string s_opname = "neq";
        NeqOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs) : CmpOp(descriptor, lhs, rhs) {}
        Opcode get_opcode() const override { return Opcode::NEQ; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct LtOp : public CmpOp {
    public:
        inline static const std::string s_opname = "lt";
        LtOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs) : CmpOp(descriptor, lhs, rhs) {}
        Opcode get_opcode() const override { return Opcode::LT; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct GtOp : public CmpOp {
    public:
        inline static const std::string s_opname = "gt";
        GtOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs) : CmpOp(descriptor, lhs, rhs) {}
        Opcode get_opcode() const override { return Opcode::GT; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct LeqOp : public CmpOp {
    public:
        inline static const std::string s_opname = "leq";
        LeqOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs) : CmpOp(descriptor, lhs, rhs) {}
        Opcode get_opcode() const override { return Opcode::LEQ; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct GeqOp : public CmpOp {
    public:
        inline static const std::string s_opname = "geq";
        GeqOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs) : CmpOp(descriptor, lhs, rhs) {}
        Opcode get_opcode() const override { return Opcode::GEQ; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct MinimumOp : public ElmwiseBinaryOp {
    public:
        inline static const std::string s_opname = "minimum";
        MinimumOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(descriptor, lhs, rhs, in_place) {}
        Opcode get_opcode() const override { return Opcode::MINIMUM; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct MaximumOp : public ElmwiseBinaryOp {
    public:
        inline static const std::string s_opname = "maximum";
        MaximumOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(descriptor, lhs, rhs, in_place) {}
        Opcode get_opcode() const override { return Opcode::MAXIMUM; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct MatmulOp : public BinaryOp {
    public:
        inline static const std::string s_opname = "matmul";
        MatmulOp(const ArrayDescriptor &descriptor, OpPtr lhs, OpPtr rhs) : BinaryOp(descriptor, lhs, rhs) {}
        Opcode get_opcode() const override { return Opcode::MATMUL; }
        BinaryMode get_mode() const override { return BinaryMode::MATMUL; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct SqOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "sq";
        SqOp(const ArrayDescriptor &descriptor, OpPtr operand, bool in_place) : UnaryOp(descriptor, operand, in_place) {}
        Opcode get_opcode() const override { return Opcode::SQ; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct SqrtOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "sqrt";
        SqrtOp(const ArrayDescriptor &descriptor, OpPtr operand, bool in_place) : UnaryOp(descriptor, operand, in_place) {}
        Opcode get_opcode() const override { return Opcode::SQRT; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct NegOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "neg";
        NegOp(const ArrayDescriptor &descriptor, OpPtr operand, bool in_place) : UnaryOp(descriptor, operand, in_place) {}
        Opcode get_opcode() const override { return Opcode::NEG; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct CopyOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "copy";
        CopyOp(const ArrayDescriptor &descriptor, OpPtr operand) : UnaryOp(descriptor, operand, false) {}
        Opcode get_opcode() const override { return Opcode::COPY; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct ExpOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "exp";
        ExpOp(const ArrayDescriptor &descriptor, OpPtr operand, bool in_place) : UnaryOp(descriptor, operand, in_place) {}
        Opcode get_opcode() const override { return Opcode::EXP; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct LogOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "log";
        LogOp(const ArrayDescriptor &descriptor, OpPtr operand, bool in_place) : UnaryOp(descriptor, operand, in_place) {}
        Opcode get_opcode() const override { return Opcode::LOG; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct RecipOp : public UnaryOp {
    public:
        inline static const std::string s_opname = "recip";
        RecipOp(const ArrayDescriptor &descriptor, OpPtr operand, bool in_place) : UnaryOp(descriptor, operand, in_place) {}
        Opcode get_opcode() const override { return Opcode::RECIP; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct ReshapeOp : public TransformOp {
    private:
        ShapeView m_view;

    public:
        inline static const std::string s_opname = "reshape";
        ReshapeOp(const ArrayDescriptor &descriptor, OpPtr operand, const ShapeView &view) : TransformOp(descriptor, operand), m_view(view) {}
        const ShapeView &get_view() const { return m_view; }
        Opcode get_opcode() const override { return Opcode::RESHAPE; }
        const std::string &get_opname() const override { return s_opname; }
        const std::string str() const override { return std::format("{}, view: ({})", TransformOp::str(), join_nums(m_view)); }
    };

    struct SliceOp : public TransformOp {
    private:
        RangeVec m_ranges;

    public:
        inline static const std::string s_opname = "slice";
        SliceOp(const ArrayDescriptor &descriptor, OpPtr operand, const RangeVec &ranges) : TransformOp(descriptor, operand), m_ranges(ranges) {}
        const RangeVec &get_ranges() const { return m_ranges; }
        Opcode get_opcode() const override { return Opcode::SLICE; }
        const std::string &get_opname() const override { return s_opname; }

        const std::string str() const override {
            return std::format("{}, ranges: ({})", TransformOp::str(), join<Range>(m_ranges, [](Range range) { return range.str(); }));
        }
    };

    struct PermuteOp : public TransformOp {
    private:
        ShapeDims m_dims;

    public:
        inline static const std::string s_opname = "permute";
        PermuteOp(const ArrayDescriptor &descriptor, OpPtr operand, const ShapeDims &dims) : TransformOp(descriptor, operand), m_dims(dims) {}
        const ShapeDims &get_perm() const { return m_dims; }
        Opcode get_opcode() const override { return Opcode::PERMUTE; }
        const std::string &get_opname() const override { return s_opname; }
        const std::string str() const override { return std::format("{}, permutation: ({})", TransformOp::str(), join_nums(m_dims)); }
    };

    struct BroadcastOp : public TransformOp {
    private:
        ShapeView m_input_view;
        ShapeView m_output_view;
        ShapeDims m_dims;

    public:
        inline static const std::string s_opname = "broadcast";
        BroadcastOp(const ArrayDescriptor &descriptor, OpPtr operand, const ShapeView &input_view, const ShapeView &output_view, const ShapeDims &dims) : TransformOp(descriptor, operand), m_input_view(input_view), m_output_view(output_view), m_dims(dims) {}
        const ShapeView &get_input_view() const { return m_input_view; }
        const ShapeView &get_output_view() const { return m_output_view; }
        const ShapeDims &get_dims() const { return m_dims; }
        Opcode get_opcode() const override { return Opcode::BROADCAST; }
        const std::string &get_opname() const override { return s_opname; }
        const std::string str() const override { return std::format("{}, input view: ({}), output view: ({})", TransformOp::str(), join_nums(m_input_view), join_nums(m_output_view)); }
    };

    struct SqueezeOp : public TransformOp {
    private:
        ShapeDims m_dims;

    public:
        inline static const std::string s_opname = "squeeze";
        SqueezeOp(const ArrayDescriptor &descriptor, OpPtr operand, const ShapeDims &dims) : TransformOp(descriptor, operand), m_dims(dims) {}
        const ShapeDims &get_dims() const { return m_dims; }
        Opcode get_opcode() const override { return Opcode::SQUEEZE; }
        const std::string &get_opname() const override { return s_opname; }
        const std::string str() const override { return std::format("{}, dims: ({})", TransformOp::str(), join_nums(m_dims)); }
    };

    struct UnsqueezeOp : public TransformOp {
    private:
        ShapeDims m_dims;

    public:
        inline static const std::string s_opname = "unsqueeze";
        UnsqueezeOp(const ArrayDescriptor &descriptor, OpPtr operand, const ShapeDims &dims) : TransformOp(descriptor, operand), m_dims(dims) {}
        const ShapeDims &get_dims() const { return m_dims; }
        Opcode get_opcode() const override { return Opcode::UNSQUEEZE; }
        const std::string &get_opname() const override { return s_opname; }
        const std::string str() const override { return std::format("{}, dims: ({})", TransformOp::str(), join_nums(m_dims)); }
    };

    struct AstypeOp : public TransformOp {
    private:
        DtypePtr m_dtype;

    public:
        inline static const std::string s_opname = "astype";
        AstypeOp(const ArrayDescriptor &descriptor, OpPtr operand, DtypePtr dtype) : TransformOp(descriptor, operand), m_dtype(dtype) {}
        DtypePtr get_dtype() const { return m_dtype; }
        Opcode get_opcode() const override { return Opcode::ASTYPE; }
        const std::string &get_opname() const override { return s_opname; }
        void enable_grad(bool enabled) override { m_grad_enabled = false; }
        const std::string str() const override { return std::format("{}, dtype: {}", TransformOp::str(), m_dtype->str()); }
    };

    struct SumOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "sum";
        SumOp(const ArrayDescriptor &descriptor, OpPtr operand, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) : ReduceOp(descriptor, operand, remaining_dims, reduce_dims) {}
        Opcode get_opcode() const override { return Opcode::SUM; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct MaxOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "max";
        MaxOp(const ArrayDescriptor &descriptor, OpPtr operand, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) : ReduceOp(descriptor, operand, remaining_dims, reduce_dims) {}
        Opcode get_opcode() const override { return Opcode::MAX; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct MinOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "min";
        MinOp(const ArrayDescriptor &descriptor, OpPtr operand, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) : ReduceOp(descriptor, operand, remaining_dims, reduce_dims) {}
        Opcode get_opcode() const override { return Opcode::MIN; }
        const std::string &get_opname() const override { return s_opname; }
    };

    struct ArgmaxOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "argmax";
        ArgmaxOp(const ArrayDescriptor &descriptor, OpPtr operand, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) : ReduceOp(descriptor, operand, remaining_dims, reduce_dims) {}
        Opcode get_opcode() const override { return Opcode::ARGMAX; }
        const std::string &get_opname() const override { return s_opname; }
        void enable_grad(bool enabled) override { m_grad_enabled = false; }
    };

    struct ArgminOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "argmin";
        ArgminOp(const ArrayDescriptor &descriptor, OpPtr operand, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) : ReduceOp(descriptor, operand, remaining_dims, reduce_dims) {}
        Opcode get_opcode() const override { return Opcode::ARGMIN; }
        const std::string &get_opname() const override { return s_opname; }
        void enable_grad(bool enabled) override { m_grad_enabled = false; }
    };
} // namespace nx::core