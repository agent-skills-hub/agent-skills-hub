import XCTest
@testable import <#AppModule#>

final class <#FeatureName#>Tests: XCTestCase {
    func test_when<#Condition#>_then<#ExpectedResult#>() async throws {
        // Arrange
        let service = MockService()
        let repository = MockRepository(service: service)
        let sut = <#FeatureOwner#>(repository: repository)

        // Act
        try await sut.<#action#>()

        // Assert
        XCTAssertEqual(<#actual#>, <#expected#>)
    }
}

private struct MockService {
    // TEST-ONLY: Fill with deterministic behavior for the selected feature.
}

private struct MockRepository {
    let service: MockService
}
