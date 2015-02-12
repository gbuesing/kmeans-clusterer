require 'stopwords'
require 'fast_stemmer'

class Bag
  attr_reader :term_index, :doc_hashes, :doc_count, :doc_frequency

  DEFAULT_OPTS = {
    tf: :raw,
    idf: false,
    stopwords: :en,
    stem: true,
    min_term_length: 3
  }

  def initialize opts = {}
    @opts = DEFAULT_OPTS.merge opts
    @index = 0
    @doc_count = 0
    @term_index = Hash.new do |hsh, key|
      val = hsh[key] = @index
      @index += 1
      val
    end
    @doc_frequency = Hash.new(0)
    @stopwords = Stopwords::Snowball::Filter.new(@opts[:stopwords]) if @opts[:stopwords]
    @doc_hashes = []
  end

  def terms_count
    @index
  end

  def add_docs docs
    docs.each {|doc| add_doc(doc) }
  end

  def add_doc doc
    @doc_count += 1
    terms = extract_terms doc
    doc_hash = create_doc_hash terms
    update_doc_frequency doc_hash
    @doc_hashes << doc_hash
  end

  def to_a
    weight_tf_values!

    @doc_hashes.map do |doc_hash|
      vec = Array.new(@index, 0)
      doc_hash.each do |k, v|
        vec[k] = v
      end
      vec
    end
  end

  private

    def weight_tf_values!
      return if [:raw, :binary].include?(@opts[:tf]) && !@opts[:idf]

      @doc_hashes.each do |doc_hash|
        doc_hash.each do |k, v|
          tf = calculate_tf v, doc_hash
          if @opts[:idf]
            idf = calculate_idf k
            tf *= idf
          end
          doc_hash[k] = tf
        end
      end
    end

    def extract_terms doc
      terms = doc.downcase.gsub(/(\d|\s|\W)+/, ' ').strip.split(/\s/)
      terms.reject! {|t| t.length < @opts[:min_term_length]} if @opts[:min_term_length]
      terms = @stopwords.filter(terms) if @stopwords
      terms.map! {|t| t.stem} if @opts[:stem]
      terms
    end

    def create_doc_hash terms
      terms.uniq! if @opts[:tf] == :binary
      out = terms.inject(Hash.new(0)) do |hsh, term| 
        index = @term_index[term]
        hsh[index] +=1
        hsh
      end
    end

    def update_doc_frequency doc_hash
      doc_hash.each_key do |k|
        @doc_frequency[k] += 1
      end
    end

    def calculate_tf raw_tf, doc_hash
      case @opts[:tf]
      when :log
        1 + Math.log(raw_tf)
      when :augmented
        max_tf = doc_hash.values.max
        0.5 + (0.5 * raw_tf) / max_tf.to_f
      when :binary
        1
      else
        raw_tf
      end
    end

    def calculate_idf term_id
      Math.log (@doc_count / @doc_frequency[term_id].to_f)
    end
end
