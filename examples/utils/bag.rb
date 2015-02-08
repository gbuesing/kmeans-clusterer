require 'rubygems'
require 'bundler/setup'
require 'stopwords'
require 'fast_stemmer'

class Bag
  attr_reader :dict, :sparse_hashes

  def initialize
    @index = 0
    @dict = Hash.new do |hsh, key|
      val = hsh[key] = @index
      @index += 1
      val
    end
    @stopwords = Stopwords::Snowball::Filter.new "en"
    @sparse_hashes = []
  end

  def total_features
    @index
  end

  def << doc
    words = doc_words doc
    doc_hash = create_doc_hash words
    @sparse_hashes << doc_hash
  end

  def to_a
    @sparse_hashes.map do |doc_hash|
      vec = Array.new(@index, 0)
      doc_hash.each do |k, v|
        vec[k] = v
      end
      vec
    end
  end

  private

    def doc_words doc
      words = doc.downcase.strip.squeeze(' ').split(/\W+/)
      words = @stopwords.filter words
      words.map! {|w| w.stem}
      words.reject! {|w| w.length < 3}
      words
    end

    def create_doc_hash words
      out = words.inject(Hash.new(0)) do |hsh, w| 
        index = @dict[w]
        hsh[index] +=1
        hsh
      end
      out.freeze
    end
end
