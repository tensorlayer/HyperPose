#pragma once
#include <map>
#include <vector>

class BodyPart
{
  public:
    BodyPart() {}

    BodyPart(int part_idx, float x, float y, float score)
        : part_idx_(part_idx), x_(x), y_(y), score_(score)
    {
    }
    float x() const { return x_; }
    float y() const { return y_; }
    float score() const { return score_; }

  private:
    int part_idx_;
    float x_;
    float y_;
    float score_;
};

class Human
{
  public:
    Human() {}
    bool empty() const { return body_parts_.empty(); }

    void add(int part_idx, const BodyPart &body_parts)
    {
        body_parts_[part_idx] = body_parts;
    }

    void set_score(float score) { score_ = score; }

    bool has_part(int part_idx) const
    {
        return body_parts_.count(part_idx) > 0;
    }

    BodyPart get_part(int part_idx) const
    {
        const auto pos = body_parts_.find(part_idx);
        if (pos != body_parts_.end()) { return pos->second; }
        BodyPart bp;
        return bp;
    }

    void print() const
    {
        // for (const auto &[part_idx, body_part] : body_parts_) {
        for (const auto &it : body_parts_) {
            const auto part_idx = it.first;
            const auto body_part = it.second;
            printf("BodyPart:%d-(%.2f, %.2f) score=%.2f ", part_idx,
                   body_part.x(), body_part.y(), body_part.score());
        }
        printf("score=%.2f\n", score_);
    }

  private:
    std::map<int, BodyPart> body_parts_;
    float score_;
};
