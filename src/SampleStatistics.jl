"Store statistics as columns of data with corresponding labels."
struct SampleStatistics
    data::Matrix{Float64}
    labels::Vector{Any}
end

function Base.copy(in::SampleStatistics)
    SampleStatistics(Base.copy(in.data), Base.copy(in.labels))
end
